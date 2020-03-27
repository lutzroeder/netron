/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $util = $protobuf.util;
    
    var $root = $protobuf.roots.tf || ($protobuf.roots.tf = {});
    
    $root.tensorflow = (function() {
    
        var tensorflow = {};
    
        tensorflow.SavedModel = (function() {
    
            function SavedModel(properties) {
                this.meta_graphs = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedModel.prototype.saved_model_schema_version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            SavedModel.prototype.meta_graphs = $util.emptyArray;
    
            SavedModel.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedModel();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.saved_model_schema_version = reader.int64();
                        break;
                    case 2:
                        if (!(message.meta_graphs && message.meta_graphs.length))
                            message.meta_graphs = [];
                        message.meta_graphs.push($root.tensorflow.MetaGraphDef.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SavedModel.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SavedModel();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "saved_model_schema_version":
                        message.saved_model_schema_version = reader.int64();
                        break;
                    case "meta_graphs":
                        if (!(message.meta_graphs && message.meta_graphs.length))
                            message.meta_graphs = [];
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
    
            function MetaGraphDef(properties) {
                this.collection_def = {};
                this.signature_def = {};
                this.asset_file_def = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            MetaGraphDef.prototype.meta_info_def = null;
            MetaGraphDef.prototype.graph_def = null;
            MetaGraphDef.prototype.saver_def = null;
            MetaGraphDef.prototype.collection_def = $util.emptyObject;
            MetaGraphDef.prototype.signature_def = $util.emptyObject;
            MetaGraphDef.prototype.asset_file_def = $util.emptyArray;
            MetaGraphDef.prototype.object_graph_def = null;
    
            MetaGraphDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.MetaGraphDef(), key;
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
                        reader.skip().pos++;
                        if (message.collection_def === $util.emptyObject)
                            message.collection_def = {};
                        key = reader.string();
                        reader.pos++;
                        message.collection_def[key] = $root.tensorflow.CollectionDef.decode(reader, reader.uint32());
                        break;
                    case 5:
                        reader.skip().pos++;
                        if (message.signature_def === $util.emptyObject)
                            message.signature_def = {};
                        key = reader.string();
                        reader.pos++;
                        message.signature_def[key] = $root.tensorflow.SignatureDef.decode(reader, reader.uint32());
                        break;
                    case 6:
                        if (!(message.asset_file_def && message.asset_file_def.length))
                            message.asset_file_def = [];
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
    
            MetaGraphDef.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.MetaGraphDef(), key, value;
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                        if (message.collection_def === $util.emptyObject)
                            message.collection_def = {};
                        reader.start();
                        key = "";
                        value = null;
                        while (!reader.end())
                            switch (reader.tag()) {
                            case "key":
                                key = reader.string();
                                break;
                            case "value":
                                value = $root.tensorflow.CollectionDef.decodeText(reader, true);
                                break;
                            }
                        message.collection_def[key] = value;
                        break;
                    case "signature_def":
                        if (message.signature_def === $util.emptyObject)
                            message.signature_def = {};
                        reader.start();
                        key = "";
                        value = null;
                        while (!reader.end())
                            switch (reader.tag()) {
                            case "key":
                                key = reader.string();
                                break;
                            case "value":
                                value = $root.tensorflow.SignatureDef.decodeText(reader, true);
                                break;
                            }
                        message.signature_def[key] = value;
                        break;
                    case "asset_file_def":
                        if (!(message.asset_file_def && message.asset_file_def.length))
                            message.asset_file_def = [];
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
    
                function MetaInfoDef(properties) {
                    this.tags = [];
                    this.function_aliases = {};
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                MetaInfoDef.prototype.meta_graph_version = "";
                MetaInfoDef.prototype.stripped_op_list = null;
                MetaInfoDef.prototype.any_info = null;
                MetaInfoDef.prototype.tags = $util.emptyArray;
                MetaInfoDef.prototype.tensorflow_version = "";
                MetaInfoDef.prototype.tensorflow_git_version = "";
                MetaInfoDef.prototype.stripped_default_attrs = false;
                MetaInfoDef.prototype.function_aliases = $util.emptyObject;
    
                MetaInfoDef.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.MetaGraphDef.MetaInfoDef(), key;
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                            if (!(message.tags && message.tags.length))
                                message.tags = [];
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
                            reader.skip().pos++;
                            if (message.function_aliases === $util.emptyObject)
                                message.function_aliases = {};
                            key = reader.string();
                            reader.pos++;
                            message.function_aliases[key] = reader.string();
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                MetaInfoDef.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.MetaGraphDef.MetaInfoDef(), key, value;
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
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
                            if (!(message.tags && message.tags.length))
                                message.tags = [];
                            if (reader.first())
                                while (!reader.last()) {
                                    message.tags.push(reader.string());
                                    reader.next();
                                }
                            else
                                message.tags.push(reader.string());
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
                            if (message.function_aliases === $util.emptyObject)
                                message.function_aliases = {};
                            reader.start();
                            key = "";
                            value = "";
                            while (!reader.end())
                                switch (reader.tag()) {
                                case "key":
                                    key = reader.string();
                                    break;
                                case "value":
                                    value = reader.string();
                                    break;
                                }
                            message.function_aliases[key] = value;
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
    
            function CollectionDef(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            CollectionDef.prototype.node_list = null;
            CollectionDef.prototype.bytes_list = null;
            CollectionDef.prototype.int64_list = null;
            CollectionDef.prototype.float_list = null;
            CollectionDef.prototype.any_list = null;
    
            var $oneOfFields;
    
            Object.defineProperty(CollectionDef.prototype, "kind", {
                get: $util.oneOfGetter($oneOfFields = ["node_list", "bytes_list", "int64_list", "float_list", "any_list"]),
                set: $util.oneOfSetter($oneOfFields)
            });
    
            CollectionDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.CollectionDef();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            CollectionDef.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.CollectionDef();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
                function NodeList(properties) {
                    this.value = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                NodeList.prototype.value = $util.emptyArray;
    
                NodeList.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.CollectionDef.NodeList();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.value && message.value.length))
                                message.value = [];
                            message.value.push(reader.string());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                NodeList.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.CollectionDef.NodeList();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "value":
                            if (!(message.value && message.value.length))
                                message.value = [];
                            if (reader.first())
                                while (!reader.last()) {
                                    message.value.push(reader.string());
                                    reader.next();
                                }
                            else
                                message.value.push(reader.string());
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
    
                function BytesList(properties) {
                    this.value = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                BytesList.prototype.value = $util.emptyArray;
    
                BytesList.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.CollectionDef.BytesList();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.value && message.value.length))
                                message.value = [];
                            message.value.push(reader.bytes());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                BytesList.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.CollectionDef.BytesList();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "value":
                            if (!(message.value && message.value.length))
                                message.value = [];
                            if (reader.first())
                                while (!reader.last()) {
                                    message.value.push(reader.bytes());
                                    reader.next();
                                }
                            else
                                message.value.push(reader.bytes());
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
    
                function Int64List(properties) {
                    this.value = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Int64List.prototype.value = $util.emptyArray;
    
                Int64List.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.CollectionDef.Int64List();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.value && message.value.length))
                                message.value = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.value.push(reader.int64());
                            } else
                                message.value.push(reader.int64());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                Int64List.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.CollectionDef.Int64List();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "value":
                            if (!(message.value && message.value.length))
                                message.value = [];
                            if (reader.first())
                                while (!reader.last()) {
                                    message.value.push(reader.int64());
                                    reader.next();
                                }
                            else
                                message.value.push(reader.int64());
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
    
                function FloatList(properties) {
                    this.value = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                FloatList.prototype.value = $util.emptyArray;
    
                FloatList.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.CollectionDef.FloatList();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.value && message.value.length))
                                message.value = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.value.push(reader.float());
                            } else
                                message.value.push(reader.float());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                FloatList.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.CollectionDef.FloatList();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "value":
                            if (!(message.value && message.value.length))
                                message.value = [];
                            if (reader.first())
                                while (!reader.last()) {
                                    message.value.push(reader.float());
                                    reader.next();
                                }
                            else
                                message.value.push(reader.float());
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
    
                function AnyList(properties) {
                    this.value = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                AnyList.prototype.value = $util.emptyArray;
    
                AnyList.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.CollectionDef.AnyList();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.value && message.value.length))
                                message.value = [];
                            message.value.push($root.google.protobuf.Any.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                AnyList.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.CollectionDef.AnyList();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "value":
                            if (!(message.value && message.value.length))
                                message.value = [];
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
    
            function TensorInfo(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorInfo.prototype.name = "";
            TensorInfo.prototype.coo_sparse = null;
            TensorInfo.prototype.composite_tensor = null;
            TensorInfo.prototype.dtype = 0;
            TensorInfo.prototype.tensor_shape = null;
    
            var $oneOfFields;
    
            Object.defineProperty(TensorInfo.prototype, "encoding", {
                get: $util.oneOfGetter($oneOfFields = ["name", "coo_sparse", "composite_tensor"]),
                set: $util.oneOfSetter($oneOfFields)
            });
    
            TensorInfo.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TensorInfo();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            TensorInfo.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.TensorInfo();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
                function CooSparse(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                CooSparse.prototype.values_tensor_name = "";
                CooSparse.prototype.indices_tensor_name = "";
                CooSparse.prototype.dense_shape_tensor_name = "";
    
                CooSparse.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TensorInfo.CooSparse();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                CooSparse.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.TensorInfo.CooSparse();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
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
    
                function CompositeTensor(properties) {
                    this.components = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                CompositeTensor.prototype.type_spec = null;
                CompositeTensor.prototype.components = $util.emptyArray;
    
                CompositeTensor.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TensorInfo.CompositeTensor();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.type_spec = $root.tensorflow.TypeSpecProto.decode(reader, reader.uint32());
                            break;
                        case 2:
                            if (!(message.components && message.components.length))
                                message.components = [];
                            message.components.push($root.tensorflow.TensorInfo.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                CompositeTensor.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.TensorInfo.CompositeTensor();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "type_spec":
                            message.type_spec = $root.tensorflow.TypeSpecProto.decodeText(reader, true);
                            break;
                        case "components":
                            if (!(message.components && message.components.length))
                                message.components = [];
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
    
            function SignatureDef(properties) {
                this.inputs = {};
                this.outputs = {};
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SignatureDef.prototype.inputs = $util.emptyObject;
            SignatureDef.prototype.outputs = $util.emptyObject;
            SignatureDef.prototype.method_name = "";
    
            SignatureDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SignatureDef(), key;
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        reader.skip().pos++;
                        if (message.inputs === $util.emptyObject)
                            message.inputs = {};
                        key = reader.string();
                        reader.pos++;
                        message.inputs[key] = $root.tensorflow.TensorInfo.decode(reader, reader.uint32());
                        break;
                    case 2:
                        reader.skip().pos++;
                        if (message.outputs === $util.emptyObject)
                            message.outputs = {};
                        key = reader.string();
                        reader.pos++;
                        message.outputs[key] = $root.tensorflow.TensorInfo.decode(reader, reader.uint32());
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
    
            SignatureDef.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SignatureDef(), key, value;
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "inputs":
                        if (message.inputs === $util.emptyObject)
                            message.inputs = {};
                        reader.start();
                        key = "";
                        value = null;
                        while (!reader.end())
                            switch (reader.tag()) {
                            case "key":
                                key = reader.string();
                                break;
                            case "value":
                                value = $root.tensorflow.TensorInfo.decodeText(reader, true);
                                break;
                            }
                        message.inputs[key] = value;
                        break;
                    case "outputs":
                        if (message.outputs === $util.emptyObject)
                            message.outputs = {};
                        reader.start();
                        key = "";
                        value = null;
                        while (!reader.end())
                            switch (reader.tag()) {
                            case "key":
                                key = reader.string();
                                break;
                            case "value":
                                value = $root.tensorflow.TensorInfo.decodeText(reader, true);
                                break;
                            }
                        message.outputs[key] = value;
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
    
            function AssetFileDef(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            AssetFileDef.prototype.tensor_info = null;
            AssetFileDef.prototype.filename = "";
    
            AssetFileDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.AssetFileDef();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            AssetFileDef.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.AssetFileDef();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
        tensorflow.SaverDef = (function() {
    
            function SaverDef(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SaverDef.prototype.filename_tensor_name = "";
            SaverDef.prototype.save_tensor_name = "";
            SaverDef.prototype.restore_op_name = "";
            SaverDef.prototype.max_to_keep = 0;
            SaverDef.prototype.sharded = false;
            SaverDef.prototype.keep_checkpoint_every_n_hours = 0;
            SaverDef.prototype.version = 0;
    
            SaverDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SaverDef();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            SaverDef.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SaverDef();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "LEGACY"] = 0;
                values[valuesById[1] = "V1"] = 1;
                values[valuesById[2] = "V2"] = 2;
                return values;
            })();
    
            return SaverDef;
        })();
    
        tensorflow.GraphDef = (function() {
    
            function GraphDef(properties) {
                this.node = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            GraphDef.prototype.node = $util.emptyArray;
            GraphDef.prototype.versions = null;
            GraphDef.prototype.version = 0;
            GraphDef.prototype.library = null;
    
            GraphDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.GraphDef();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.node && message.node.length))
                            message.node = [];
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
    
            GraphDef.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.GraphDef();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "node":
                        if (!(message.node && message.node.length))
                            message.node = [];
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
    
        tensorflow.OpDef = (function() {
    
            function OpDef(properties) {
                this.input_arg = [];
                this.output_arg = [];
                this.control_output = [];
                this.attr = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            OpDef.prototype.name = "";
            OpDef.prototype.input_arg = $util.emptyArray;
            OpDef.prototype.output_arg = $util.emptyArray;
            OpDef.prototype.control_output = $util.emptyArray;
            OpDef.prototype.attr = $util.emptyArray;
            OpDef.prototype.deprecation = null;
            OpDef.prototype.summary = "";
            OpDef.prototype.description = "";
            OpDef.prototype.is_commutative = false;
            OpDef.prototype.is_aggregate = false;
            OpDef.prototype.is_stateful = false;
            OpDef.prototype.allows_uninitialized_input = false;
    
            OpDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.OpDef();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        if (!(message.input_arg && message.input_arg.length))
                            message.input_arg = [];
                        message.input_arg.push($root.tensorflow.OpDef.ArgDef.decode(reader, reader.uint32()));
                        break;
                    case 3:
                        if (!(message.output_arg && message.output_arg.length))
                            message.output_arg = [];
                        message.output_arg.push($root.tensorflow.OpDef.ArgDef.decode(reader, reader.uint32()));
                        break;
                    case 20:
                        if (!(message.control_output && message.control_output.length))
                            message.control_output = [];
                        message.control_output.push(reader.string());
                        break;
                    case 4:
                        if (!(message.attr && message.attr.length))
                            message.attr = [];
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
    
            OpDef.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.OpDef();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "input_arg":
                        if (!(message.input_arg && message.input_arg.length))
                            message.input_arg = [];
                        message.input_arg.push($root.tensorflow.OpDef.ArgDef.decodeText(reader, true));
                        break;
                    case "output_arg":
                        if (!(message.output_arg && message.output_arg.length))
                            message.output_arg = [];
                        message.output_arg.push($root.tensorflow.OpDef.ArgDef.decodeText(reader, true));
                        break;
                    case "control_output":
                        if (!(message.control_output && message.control_output.length))
                            message.control_output = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.control_output.push(reader.string());
                                reader.next();
                            }
                        else
                            message.control_output.push(reader.string());
                        break;
                    case "attr":
                        if (!(message.attr && message.attr.length))
                            message.attr = [];
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
    
                function ArgDef(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ArgDef.prototype.name = "";
                ArgDef.prototype.description = "";
                ArgDef.prototype.type = 0;
                ArgDef.prototype.type_attr = "";
                ArgDef.prototype.number_attr = "";
                ArgDef.prototype.type_list_attr = "";
                ArgDef.prototype.is_ref = false;
    
                ArgDef.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.OpDef.ArgDef();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                ArgDef.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.OpDef.ArgDef();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
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
    
                function AttrDef(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                AttrDef.prototype.name = "";
                AttrDef.prototype.type = "";
                AttrDef.prototype.default_value = null;
                AttrDef.prototype.description = "";
                AttrDef.prototype.has_minimum = false;
                AttrDef.prototype.minimum = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                AttrDef.prototype.allowed_values = null;
    
                AttrDef.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.OpDef.AttrDef();
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
    
                AttrDef.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.OpDef.AttrDef();
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
    
            function OpDeprecation(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            OpDeprecation.prototype.version = 0;
            OpDeprecation.prototype.explanation = "";
    
            OpDeprecation.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.OpDeprecation();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            OpDeprecation.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.OpDeprecation();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function OpList(properties) {
                this.op = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            OpList.prototype.op = $util.emptyArray;
    
            OpList.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.OpList();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.op && message.op.length))
                            message.op = [];
                        message.op.push($root.tensorflow.OpDef.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            OpList.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.OpList();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "op":
                        if (!(message.op && message.op.length))
                            message.op = [];
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
    
        tensorflow.TensorShapeProto = (function() {
    
            function TensorShapeProto(properties) {
                this.dim = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorShapeProto.prototype.dim = $util.emptyArray;
            TensorShapeProto.prototype.unknown_rank = false;
    
            TensorShapeProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TensorShapeProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 2:
                        if (!(message.dim && message.dim.length))
                            message.dim = [];
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
    
            TensorShapeProto.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.TensorShapeProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "dim":
                        if (!(message.dim && message.dim.length))
                            message.dim = [];
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
    
                function Dim(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Dim.prototype.size = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                Dim.prototype.name = "";
    
                Dim.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TensorShapeProto.Dim();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                Dim.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.TensorShapeProto.Dim();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
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
            var valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "DT_INVALID"] = 0;
            values[valuesById[1] = "DT_FLOAT"] = 1;
            values[valuesById[2] = "DT_DOUBLE"] = 2;
            values[valuesById[3] = "DT_INT32"] = 3;
            values[valuesById[4] = "DT_UINT8"] = 4;
            values[valuesById[5] = "DT_INT16"] = 5;
            values[valuesById[6] = "DT_INT8"] = 6;
            values[valuesById[7] = "DT_STRING"] = 7;
            values[valuesById[8] = "DT_COMPLEX64"] = 8;
            values[valuesById[9] = "DT_INT64"] = 9;
            values[valuesById[10] = "DT_BOOL"] = 10;
            values[valuesById[11] = "DT_QINT8"] = 11;
            values[valuesById[12] = "DT_QUINT8"] = 12;
            values[valuesById[13] = "DT_QINT32"] = 13;
            values[valuesById[14] = "DT_BFLOAT16"] = 14;
            values[valuesById[15] = "DT_QINT16"] = 15;
            values[valuesById[16] = "DT_QUINT16"] = 16;
            values[valuesById[17] = "DT_UINT16"] = 17;
            values[valuesById[18] = "DT_COMPLEX128"] = 18;
            values[valuesById[19] = "DT_HALF"] = 19;
            values[valuesById[20] = "DT_RESOURCE"] = 20;
            values[valuesById[21] = "DT_VARIANT"] = 21;
            values[valuesById[22] = "DT_UINT32"] = 22;
            values[valuesById[23] = "DT_UINT64"] = 23;
            values[valuesById[101] = "DT_FLOAT_REF"] = 101;
            values[valuesById[102] = "DT_DOUBLE_REF"] = 102;
            values[valuesById[103] = "DT_INT32_REF"] = 103;
            values[valuesById[104] = "DT_UINT8_REF"] = 104;
            values[valuesById[105] = "DT_INT16_REF"] = 105;
            values[valuesById[106] = "DT_INT8_REF"] = 106;
            values[valuesById[107] = "DT_STRING_REF"] = 107;
            values[valuesById[108] = "DT_COMPLEX64_REF"] = 108;
            values[valuesById[109] = "DT_INT64_REF"] = 109;
            values[valuesById[110] = "DT_BOOL_REF"] = 110;
            values[valuesById[111] = "DT_QINT8_REF"] = 111;
            values[valuesById[112] = "DT_QUINT8_REF"] = 112;
            values[valuesById[113] = "DT_QINT32_REF"] = 113;
            values[valuesById[114] = "DT_BFLOAT16_REF"] = 114;
            values[valuesById[115] = "DT_QINT16_REF"] = 115;
            values[valuesById[116] = "DT_QUINT16_REF"] = 116;
            values[valuesById[117] = "DT_UINT16_REF"] = 117;
            values[valuesById[118] = "DT_COMPLEX128_REF"] = 118;
            values[valuesById[119] = "DT_HALF_REF"] = 119;
            values[valuesById[120] = "DT_RESOURCE_REF"] = 120;
            values[valuesById[121] = "DT_VARIANT_REF"] = 121;
            values[valuesById[122] = "DT_UINT32_REF"] = 122;
            values[valuesById[123] = "DT_UINT64_REF"] = 123;
            return values;
        })();
    
        tensorflow.NodeDef = (function() {
    
            function NodeDef(properties) {
                this.input = [];
                this.attr = {};
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            NodeDef.prototype.name = "";
            NodeDef.prototype.op = "";
            NodeDef.prototype.input = $util.emptyArray;
            NodeDef.prototype.device = "";
            NodeDef.prototype.attr = $util.emptyObject;
            NodeDef.prototype.experimental_debug_info = null;
    
            NodeDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.NodeDef(), key;
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        message.op = reader.string();
                        break;
                    case 3:
                        if (!(message.input && message.input.length))
                            message.input = [];
                        message.input.push(reader.string());
                        break;
                    case 4:
                        message.device = reader.string();
                        break;
                    case 5:
                        reader.skip().pos++;
                        if (message.attr === $util.emptyObject)
                            message.attr = {};
                        key = reader.string();
                        reader.pos++;
                        message.attr[key] = $root.tensorflow.AttrValue.decode(reader, reader.uint32());
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
    
            NodeDef.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.NodeDef(), key, value;
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "op":
                        message.op = reader.string();
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
                    case "device":
                        message.device = reader.string();
                        break;
                    case "attr":
                        if (message.attr === $util.emptyObject)
                            message.attr = {};
                        reader.start();
                        key = "";
                        value = null;
                        while (!reader.end())
                            switch (reader.tag()) {
                            case "key":
                                key = reader.string();
                                break;
                            case "value":
                                value = $root.tensorflow.AttrValue.decodeText(reader, true);
                                break;
                            }
                        message.attr[key] = value;
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
    
                function ExperimentalDebugInfo(properties) {
                    this.original_node_names = [];
                    this.original_func_names = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ExperimentalDebugInfo.prototype.original_node_names = $util.emptyArray;
                ExperimentalDebugInfo.prototype.original_func_names = $util.emptyArray;
    
                ExperimentalDebugInfo.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.NodeDef.ExperimentalDebugInfo();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.original_node_names && message.original_node_names.length))
                                message.original_node_names = [];
                            message.original_node_names.push(reader.string());
                            break;
                        case 2:
                            if (!(message.original_func_names && message.original_func_names.length))
                                message.original_func_names = [];
                            message.original_func_names.push(reader.string());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                ExperimentalDebugInfo.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.NodeDef.ExperimentalDebugInfo();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "original_node_names":
                            if (!(message.original_node_names && message.original_node_names.length))
                                message.original_node_names = [];
                            if (reader.first())
                                while (!reader.last()) {
                                    message.original_node_names.push(reader.string());
                                    reader.next();
                                }
                            else
                                message.original_node_names.push(reader.string());
                            break;
                        case "original_func_names":
                            if (!(message.original_func_names && message.original_func_names.length))
                                message.original_func_names = [];
                            if (reader.first())
                                while (!reader.last()) {
                                    message.original_func_names.push(reader.string());
                                    reader.next();
                                }
                            else
                                message.original_func_names.push(reader.string());
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
    
        tensorflow.VersionDef = (function() {
    
            function VersionDef(properties) {
                this.bad_consumers = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            VersionDef.prototype.producer = 0;
            VersionDef.prototype.min_consumer = 0;
            VersionDef.prototype.bad_consumers = $util.emptyArray;
    
            VersionDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.VersionDef();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.producer = reader.int32();
                        break;
                    case 2:
                        message.min_consumer = reader.int32();
                        break;
                    case 3:
                        if (!(message.bad_consumers && message.bad_consumers.length))
                            message.bad_consumers = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.bad_consumers.push(reader.int32());
                        } else
                            message.bad_consumers.push(reader.int32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            VersionDef.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.VersionDef();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "producer":
                        message.producer = reader.int32();
                        break;
                    case "min_consumer":
                        message.min_consumer = reader.int32();
                        break;
                    case "bad_consumers":
                        if (!(message.bad_consumers && message.bad_consumers.length))
                            message.bad_consumers = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.bad_consumers.push(reader.int32());
                                reader.next();
                            }
                        else
                            message.bad_consumers.push(reader.int32());
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
    
        tensorflow.FunctionDefLibrary = (function() {
    
            function FunctionDefLibrary(properties) {
                this["function"] = [];
                this.gradient = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            FunctionDefLibrary.prototype["function"] = $util.emptyArray;
            FunctionDefLibrary.prototype.gradient = $util.emptyArray;
    
            FunctionDefLibrary.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.FunctionDefLibrary();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message["function"] && message["function"].length))
                            message["function"] = [];
                        message["function"].push($root.tensorflow.FunctionDef.decode(reader, reader.uint32()));
                        break;
                    case 2:
                        if (!(message.gradient && message.gradient.length))
                            message.gradient = [];
                        message.gradient.push($root.tensorflow.GradientDef.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            FunctionDefLibrary.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.FunctionDefLibrary();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "function":
                        if (!(message["function"] && message["function"].length))
                            message["function"] = [];
                        message["function"].push($root.tensorflow.FunctionDef.decodeText(reader, true));
                        break;
                    case "gradient":
                        if (!(message.gradient && message.gradient.length))
                            message.gradient = [];
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
    
            function FunctionDef(properties) {
                this.attr = {};
                this.arg_attr = {};
                this.resource_arg_unique_id = {};
                this.node_def = [];
                this.ret = {};
                this.control_ret = {};
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            FunctionDef.prototype.signature = null;
            FunctionDef.prototype.attr = $util.emptyObject;
            FunctionDef.prototype.arg_attr = $util.emptyObject;
            FunctionDef.prototype.resource_arg_unique_id = $util.emptyObject;
            FunctionDef.prototype.node_def = $util.emptyArray;
            FunctionDef.prototype.ret = $util.emptyObject;
            FunctionDef.prototype.control_ret = $util.emptyObject;
    
            FunctionDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.FunctionDef(), key;
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.signature = $root.tensorflow.OpDef.decode(reader, reader.uint32());
                        break;
                    case 5:
                        reader.skip().pos++;
                        if (message.attr === $util.emptyObject)
                            message.attr = {};
                        key = reader.string();
                        reader.pos++;
                        message.attr[key] = $root.tensorflow.AttrValue.decode(reader, reader.uint32());
                        break;
                    case 7:
                        reader.skip().pos++;
                        if (message.arg_attr === $util.emptyObject)
                            message.arg_attr = {};
                        key = reader.uint32();
                        reader.pos++;
                        message.arg_attr[key] = $root.tensorflow.FunctionDef.ArgAttrs.decode(reader, reader.uint32());
                        break;
                    case 8:
                        reader.skip().pos++;
                        if (message.resource_arg_unique_id === $util.emptyObject)
                            message.resource_arg_unique_id = {};
                        key = reader.uint32();
                        reader.pos++;
                        message.resource_arg_unique_id[key] = reader.uint32();
                        break;
                    case 3:
                        if (!(message.node_def && message.node_def.length))
                            message.node_def = [];
                        message.node_def.push($root.tensorflow.NodeDef.decode(reader, reader.uint32()));
                        break;
                    case 4:
                        reader.skip().pos++;
                        if (message.ret === $util.emptyObject)
                            message.ret = {};
                        key = reader.string();
                        reader.pos++;
                        message.ret[key] = reader.string();
                        break;
                    case 6:
                        reader.skip().pos++;
                        if (message.control_ret === $util.emptyObject)
                            message.control_ret = {};
                        key = reader.string();
                        reader.pos++;
                        message.control_ret[key] = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            FunctionDef.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.FunctionDef(), key, value;
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "signature":
                        message.signature = $root.tensorflow.OpDef.decodeText(reader, true);
                        break;
                    case "attr":
                        if (message.attr === $util.emptyObject)
                            message.attr = {};
                        reader.start();
                        key = "";
                        value = null;
                        while (!reader.end())
                            switch (reader.tag()) {
                            case "key":
                                key = reader.string();
                                break;
                            case "value":
                                value = $root.tensorflow.AttrValue.decodeText(reader, true);
                                break;
                            }
                        message.attr[key] = value;
                        break;
                    case "arg_attr":
                        if (message.arg_attr === $util.emptyObject)
                            message.arg_attr = {};
                        reader.start();
                        key = 0;
                        value = null;
                        while (!reader.end())
                            switch (reader.tag()) {
                            case "key":
                                key = reader.uint32();
                                break;
                            case "value":
                                value = $root.tensorflow.FunctionDef.ArgAttrs.decodeText(reader, true);
                                break;
                            }
                        message.arg_attr[key] = value;
                        break;
                    case "resource_arg_unique_id":
                        if (message.resource_arg_unique_id === $util.emptyObject)
                            message.resource_arg_unique_id = {};
                        reader.start();
                        key = 0;
                        value = 0;
                        while (!reader.end())
                            switch (reader.tag()) {
                            case "key":
                                key = reader.uint32();
                                break;
                            case "value":
                                value = reader.uint32();
                                break;
                            }
                        message.resource_arg_unique_id[key] = value;
                        break;
                    case "node_def":
                        if (!(message.node_def && message.node_def.length))
                            message.node_def = [];
                        message.node_def.push($root.tensorflow.NodeDef.decodeText(reader, true));
                        break;
                    case "ret":
                        if (message.ret === $util.emptyObject)
                            message.ret = {};
                        reader.start();
                        key = "";
                        value = "";
                        while (!reader.end())
                            switch (reader.tag()) {
                            case "key":
                                key = reader.string();
                                break;
                            case "value":
                                value = reader.string();
                                break;
                            }
                        message.ret[key] = value;
                        break;
                    case "control_ret":
                        if (message.control_ret === $util.emptyObject)
                            message.control_ret = {};
                        reader.start();
                        key = "";
                        value = "";
                        while (!reader.end())
                            switch (reader.tag()) {
                            case "key":
                                key = reader.string();
                                break;
                            case "value":
                                value = reader.string();
                                break;
                            }
                        message.control_ret[key] = value;
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            FunctionDef.ArgAttrs = (function() {
    
                function ArgAttrs(properties) {
                    this.attr = {};
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ArgAttrs.prototype.attr = $util.emptyObject;
    
                ArgAttrs.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.FunctionDef.ArgAttrs(), key;
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            reader.skip().pos++;
                            if (message.attr === $util.emptyObject)
                                message.attr = {};
                            key = reader.string();
                            reader.pos++;
                            message.attr[key] = $root.tensorflow.AttrValue.decode(reader, reader.uint32());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                ArgAttrs.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.FunctionDef.ArgAttrs(), key, value;
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "attr":
                            if (message.attr === $util.emptyObject)
                                message.attr = {};
                            reader.start();
                            key = "";
                            value = null;
                            while (!reader.end())
                                switch (reader.tag()) {
                                case "key":
                                    key = reader.string();
                                    break;
                                case "value":
                                    value = $root.tensorflow.AttrValue.decodeText(reader, true);
                                    break;
                                }
                            message.attr[key] = value;
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
    
            function GradientDef(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            GradientDef.prototype.function_name = "";
            GradientDef.prototype.gradient_func = "";
    
            GradientDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.GradientDef();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            GradientDef.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.GradientDef();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function AttrValue(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            AttrValue.prototype.s = $util.newBuffer([]);
            AttrValue.prototype.i = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            AttrValue.prototype.f = 0;
            AttrValue.prototype.b = false;
            AttrValue.prototype.type = 0;
            AttrValue.prototype.shape = null;
            AttrValue.prototype.tensor = null;
            AttrValue.prototype.list = null;
            AttrValue.prototype.func = null;
            AttrValue.prototype.placeholder = "";
    
            var $oneOfFields;
    
            Object.defineProperty(AttrValue.prototype, "value", {
                get: $util.oneOfGetter($oneOfFields = ["s", "i", "f", "b", "type", "shape", "tensor", "list", "func", "placeholder"]),
                set: $util.oneOfSetter($oneOfFields)
            });
    
            AttrValue.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.AttrValue();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            AttrValue.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.AttrValue();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
                function ListValue(properties) {
                    this.s = [];
                    this.i = [];
                    this.f = [];
                    this.b = [];
                    this.type = [];
                    this.shape = [];
                    this.tensor = [];
                    this.func = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ListValue.prototype.s = $util.emptyArray;
                ListValue.prototype.i = $util.emptyArray;
                ListValue.prototype.f = $util.emptyArray;
                ListValue.prototype.b = $util.emptyArray;
                ListValue.prototype.type = $util.emptyArray;
                ListValue.prototype.shape = $util.emptyArray;
                ListValue.prototype.tensor = $util.emptyArray;
                ListValue.prototype.func = $util.emptyArray;
    
                ListValue.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.AttrValue.ListValue();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 2:
                            if (!(message.s && message.s.length))
                                message.s = [];
                            message.s.push(reader.bytes());
                            break;
                        case 3:
                            if (!(message.i && message.i.length))
                                message.i = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.i.push(reader.int64());
                            } else
                                message.i.push(reader.int64());
                            break;
                        case 4:
                            if (!(message.f && message.f.length))
                                message.f = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.f.push(reader.float());
                            } else
                                message.f.push(reader.float());
                            break;
                        case 5:
                            if (!(message.b && message.b.length))
                                message.b = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.b.push(reader.bool());
                            } else
                                message.b.push(reader.bool());
                            break;
                        case 6:
                            if (!(message.type && message.type.length))
                                message.type = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.type.push(reader.int32());
                            } else
                                message.type.push(reader.int32());
                            break;
                        case 7:
                            if (!(message.shape && message.shape.length))
                                message.shape = [];
                            message.shape.push($root.tensorflow.TensorShapeProto.decode(reader, reader.uint32()));
                            break;
                        case 8:
                            if (!(message.tensor && message.tensor.length))
                                message.tensor = [];
                            message.tensor.push($root.tensorflow.TensorProto.decode(reader, reader.uint32()));
                            break;
                        case 9:
                            if (!(message.func && message.func.length))
                                message.func = [];
                            message.func.push($root.tensorflow.NameAttrList.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                ListValue.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.AttrValue.ListValue();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "s":
                            if (!(message.s && message.s.length))
                                message.s = [];
                            if (reader.first())
                                while (!reader.last()) {
                                    message.s.push(reader.bytes());
                                    reader.next();
                                }
                            else
                                message.s.push(reader.bytes());
                            break;
                        case "i":
                            if (!(message.i && message.i.length))
                                message.i = [];
                            if (reader.first())
                                while (!reader.last()) {
                                    message.i.push(reader.int64());
                                    reader.next();
                                }
                            else
                                message.i.push(reader.int64());
                            break;
                        case "f":
                            if (!(message.f && message.f.length))
                                message.f = [];
                            if (reader.first())
                                while (!reader.last()) {
                                    message.f.push(reader.float());
                                    reader.next();
                                }
                            else
                                message.f.push(reader.float());
                            break;
                        case "b":
                            if (!(message.b && message.b.length))
                                message.b = [];
                            if (reader.first())
                                while (!reader.last()) {
                                    message.b.push(reader.bool());
                                    reader.next();
                                }
                            else
                                message.b.push(reader.bool());
                            break;
                        case "type":
                            if (!(message.type && message.type.length))
                                message.type = [];
                            if (reader.first())
                                while (!reader.last()) {
                                    message.type.push(reader.enum($root.tensorflow.DataType));
                                    reader.next();
                                }
                            else
                                message.type.push(reader.enum($root.tensorflow.DataType));
                            break;
                        case "shape":
                            if (!(message.shape && message.shape.length))
                                message.shape = [];
                            message.shape.push($root.tensorflow.TensorShapeProto.decodeText(reader, true));
                            break;
                        case "tensor":
                            if (!(message.tensor && message.tensor.length))
                                message.tensor = [];
                            message.tensor.push($root.tensorflow.TensorProto.decodeText(reader, true));
                            break;
                        case "func":
                            if (!(message.func && message.func.length))
                                message.func = [];
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
    
            function NameAttrList(properties) {
                this.attr = {};
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            NameAttrList.prototype.name = "";
            NameAttrList.prototype.attr = $util.emptyObject;
    
            NameAttrList.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.NameAttrList(), key;
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        reader.skip().pos++;
                        if (message.attr === $util.emptyObject)
                            message.attr = {};
                        key = reader.string();
                        reader.pos++;
                        message.attr[key] = $root.tensorflow.AttrValue.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            NameAttrList.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.NameAttrList(), key, value;
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "attr":
                        if (message.attr === $util.emptyObject)
                            message.attr = {};
                        reader.start();
                        key = "";
                        value = null;
                        while (!reader.end())
                            switch (reader.tag()) {
                            case "key":
                                key = reader.string();
                                break;
                            case "value":
                                value = $root.tensorflow.AttrValue.decodeText(reader, true);
                                break;
                            }
                        message.attr[key] = value;
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
    
            function TensorProto(properties) {
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
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorProto.prototype.dtype = 0;
            TensorProto.prototype.tensor_shape = null;
            TensorProto.prototype.version_number = 0;
            TensorProto.prototype.tensor_content = $util.newBuffer([]);
            TensorProto.prototype.half_val = $util.emptyArray;
            TensorProto.prototype.float_val = $util.emptyArray;
            TensorProto.prototype.double_val = $util.emptyArray;
            TensorProto.prototype.int_val = $util.emptyArray;
            TensorProto.prototype.string_val = $util.emptyArray;
            TensorProto.prototype.scomplex_val = $util.emptyArray;
            TensorProto.prototype.int64_val = $util.emptyArray;
            TensorProto.prototype.bool_val = $util.emptyArray;
            TensorProto.prototype.dcomplex_val = $util.emptyArray;
            TensorProto.prototype.resource_handle_val = $util.emptyArray;
            TensorProto.prototype.variant_val = $util.emptyArray;
            TensorProto.prototype.uint32_val = $util.emptyArray;
            TensorProto.prototype.uint64_val = $util.emptyArray;
    
            TensorProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TensorProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
                        if (!(message.half_val && message.half_val.length))
                            message.half_val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.half_val.push(reader.int32());
                        } else
                            message.half_val.push(reader.int32());
                        break;
                    case 5:
                        if (!(message.float_val && message.float_val.length))
                            message.float_val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.float_val.push(reader.float());
                        } else
                            message.float_val.push(reader.float());
                        break;
                    case 6:
                        if (!(message.double_val && message.double_val.length))
                            message.double_val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.double_val.push(reader.double());
                        } else
                            message.double_val.push(reader.double());
                        break;
                    case 7:
                        if (!(message.int_val && message.int_val.length))
                            message.int_val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.int_val.push(reader.int32());
                        } else
                            message.int_val.push(reader.int32());
                        break;
                    case 8:
                        if (!(message.string_val && message.string_val.length))
                            message.string_val = [];
                        message.string_val.push(reader.bytes());
                        break;
                    case 9:
                        if (!(message.scomplex_val && message.scomplex_val.length))
                            message.scomplex_val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.scomplex_val.push(reader.float());
                        } else
                            message.scomplex_val.push(reader.float());
                        break;
                    case 10:
                        if (!(message.int64_val && message.int64_val.length))
                            message.int64_val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.int64_val.push(reader.int64());
                        } else
                            message.int64_val.push(reader.int64());
                        break;
                    case 11:
                        if (!(message.bool_val && message.bool_val.length))
                            message.bool_val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.bool_val.push(reader.bool());
                        } else
                            message.bool_val.push(reader.bool());
                        break;
                    case 12:
                        if (!(message.dcomplex_val && message.dcomplex_val.length))
                            message.dcomplex_val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.dcomplex_val.push(reader.double());
                        } else
                            message.dcomplex_val.push(reader.double());
                        break;
                    case 14:
                        if (!(message.resource_handle_val && message.resource_handle_val.length))
                            message.resource_handle_val = [];
                        message.resource_handle_val.push($root.tensorflow.ResourceHandleProto.decode(reader, reader.uint32()));
                        break;
                    case 15:
                        if (!(message.variant_val && message.variant_val.length))
                            message.variant_val = [];
                        message.variant_val.push($root.tensorflow.VariantTensorDataProto.decode(reader, reader.uint32()));
                        break;
                    case 16:
                        if (!(message.uint32_val && message.uint32_val.length))
                            message.uint32_val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.uint32_val.push(reader.uint32());
                        } else
                            message.uint32_val.push(reader.uint32());
                        break;
                    case 17:
                        if (!(message.uint64_val && message.uint64_val.length))
                            message.uint64_val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.uint64_val.push(reader.uint64());
                        } else
                            message.uint64_val.push(reader.uint64());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TensorProto.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.TensorProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                        if (!(message.half_val && message.half_val.length))
                            message.half_val = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.half_val.push(reader.int32());
                                reader.next();
                            }
                        else
                            message.half_val.push(reader.int32());
                        break;
                    case "float_val":
                        if (!(message.float_val && message.float_val.length))
                            message.float_val = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.float_val.push(reader.float());
                                reader.next();
                            }
                        else
                            message.float_val.push(reader.float());
                        break;
                    case "double_val":
                        if (!(message.double_val && message.double_val.length))
                            message.double_val = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.double_val.push(reader.double());
                                reader.next();
                            }
                        else
                            message.double_val.push(reader.double());
                        break;
                    case "int_val":
                        if (!(message.int_val && message.int_val.length))
                            message.int_val = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.int_val.push(reader.int32());
                                reader.next();
                            }
                        else
                            message.int_val.push(reader.int32());
                        break;
                    case "string_val":
                        if (!(message.string_val && message.string_val.length))
                            message.string_val = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.string_val.push(reader.bytes());
                                reader.next();
                            }
                        else
                            message.string_val.push(reader.bytes());
                        break;
                    case "scomplex_val":
                        if (!(message.scomplex_val && message.scomplex_val.length))
                            message.scomplex_val = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.scomplex_val.push(reader.float());
                                reader.next();
                            }
                        else
                            message.scomplex_val.push(reader.float());
                        break;
                    case "int64_val":
                        if (!(message.int64_val && message.int64_val.length))
                            message.int64_val = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.int64_val.push(reader.int64());
                                reader.next();
                            }
                        else
                            message.int64_val.push(reader.int64());
                        break;
                    case "bool_val":
                        if (!(message.bool_val && message.bool_val.length))
                            message.bool_val = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.bool_val.push(reader.bool());
                                reader.next();
                            }
                        else
                            message.bool_val.push(reader.bool());
                        break;
                    case "dcomplex_val":
                        if (!(message.dcomplex_val && message.dcomplex_val.length))
                            message.dcomplex_val = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.dcomplex_val.push(reader.double());
                                reader.next();
                            }
                        else
                            message.dcomplex_val.push(reader.double());
                        break;
                    case "resource_handle_val":
                        if (!(message.resource_handle_val && message.resource_handle_val.length))
                            message.resource_handle_val = [];
                        message.resource_handle_val.push($root.tensorflow.ResourceHandleProto.decodeText(reader, true));
                        break;
                    case "variant_val":
                        if (!(message.variant_val && message.variant_val.length))
                            message.variant_val = [];
                        message.variant_val.push($root.tensorflow.VariantTensorDataProto.decodeText(reader, true));
                        break;
                    case "uint32_val":
                        if (!(message.uint32_val && message.uint32_val.length))
                            message.uint32_val = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.uint32_val.push(reader.uint32());
                                reader.next();
                            }
                        else
                            message.uint32_val.push(reader.uint32());
                        break;
                    case "uint64_val":
                        if (!(message.uint64_val && message.uint64_val.length))
                            message.uint64_val = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.uint64_val.push(reader.uint64());
                                reader.next();
                            }
                        else
                            message.uint64_val.push(reader.uint64());
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
    
            function VariantTensorDataProto(properties) {
                this.tensors = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            VariantTensorDataProto.prototype.type_name = "";
            VariantTensorDataProto.prototype.metadata = $util.newBuffer([]);
            VariantTensorDataProto.prototype.tensors = $util.emptyArray;
    
            VariantTensorDataProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.VariantTensorDataProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.type_name = reader.string();
                        break;
                    case 2:
                        message.metadata = reader.bytes();
                        break;
                    case 3:
                        if (!(message.tensors && message.tensors.length))
                            message.tensors = [];
                        message.tensors.push($root.tensorflow.TensorProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            VariantTensorDataProto.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.VariantTensorDataProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "type_name":
                        message.type_name = reader.string();
                        break;
                    case "metadata":
                        message.metadata = reader.bytes();
                        break;
                    case "tensors":
                        if (!(message.tensors && message.tensors.length))
                            message.tensors = [];
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
    
        tensorflow.VariableSynchronization = (function() {
            var valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "VARIABLE_SYNCHRONIZATION_AUTO"] = 0;
            values[valuesById[1] = "VARIABLE_SYNCHRONIZATION_NONE"] = 1;
            values[valuesById[2] = "VARIABLE_SYNCHRONIZATION_ON_WRITE"] = 2;
            values[valuesById[3] = "VARIABLE_SYNCHRONIZATION_ON_READ"] = 3;
            return values;
        })();
    
        tensorflow.VariableAggregation = (function() {
            var valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "VARIABLE_AGGREGATION_NONE"] = 0;
            values[valuesById[1] = "VARIABLE_AGGREGATION_SUM"] = 1;
            values[valuesById[2] = "VARIABLE_AGGREGATION_MEAN"] = 2;
            values[valuesById[3] = "VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA"] = 3;
            return values;
        })();
    
        tensorflow.VariableDef = (function() {
    
            function VariableDef(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
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
    
            VariableDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.VariableDef();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            VariableDef.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.VariableDef();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function SaveSliceInfoDef(properties) {
                this.full_shape = [];
                this.var_offset = [];
                this.var_shape = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SaveSliceInfoDef.prototype.full_name = "";
            SaveSliceInfoDef.prototype.full_shape = $util.emptyArray;
            SaveSliceInfoDef.prototype.var_offset = $util.emptyArray;
            SaveSliceInfoDef.prototype.var_shape = $util.emptyArray;
    
            SaveSliceInfoDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SaveSliceInfoDef();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.full_name = reader.string();
                        break;
                    case 2:
                        if (!(message.full_shape && message.full_shape.length))
                            message.full_shape = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.full_shape.push(reader.int64());
                        } else
                            message.full_shape.push(reader.int64());
                        break;
                    case 3:
                        if (!(message.var_offset && message.var_offset.length))
                            message.var_offset = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.var_offset.push(reader.int64());
                        } else
                            message.var_offset.push(reader.int64());
                        break;
                    case 4:
                        if (!(message.var_shape && message.var_shape.length))
                            message.var_shape = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.var_shape.push(reader.int64());
                        } else
                            message.var_shape.push(reader.int64());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SaveSliceInfoDef.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SaveSliceInfoDef();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "full_name":
                        message.full_name = reader.string();
                        break;
                    case "full_shape":
                        if (!(message.full_shape && message.full_shape.length))
                            message.full_shape = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.full_shape.push(reader.int64());
                                reader.next();
                            }
                        else
                            message.full_shape.push(reader.int64());
                        break;
                    case "var_offset":
                        if (!(message.var_offset && message.var_offset.length))
                            message.var_offset = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.var_offset.push(reader.int64());
                                reader.next();
                            }
                        else
                            message.var_offset.push(reader.int64());
                        break;
                    case "var_shape":
                        if (!(message.var_shape && message.var_shape.length))
                            message.var_shape = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.var_shape.push(reader.int64());
                                reader.next();
                            }
                        else
                            message.var_shape.push(reader.int64());
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
    
        tensorflow.ResourceHandleProto = (function() {
    
            function ResourceHandleProto(properties) {
                this.dtypes_and_shapes = [];
                this.allowed_devices = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ResourceHandleProto.prototype.device = "";
            ResourceHandleProto.prototype.container = "";
            ResourceHandleProto.prototype.name = "";
            ResourceHandleProto.prototype.hash_code = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
            ResourceHandleProto.prototype.maybe_type_name = "";
            ResourceHandleProto.prototype.dtypes_and_shapes = $util.emptyArray;
            ResourceHandleProto.prototype.allowed_devices = $util.emptyArray;
    
            ResourceHandleProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.ResourceHandleProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
                        if (!(message.dtypes_and_shapes && message.dtypes_and_shapes.length))
                            message.dtypes_and_shapes = [];
                        message.dtypes_and_shapes.push($root.tensorflow.ResourceHandleProto.DtypeAndShape.decode(reader, reader.uint32()));
                        break;
                    case 7:
                        if (!(message.allowed_devices && message.allowed_devices.length))
                            message.allowed_devices = [];
                        message.allowed_devices.push(reader.string());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ResourceHandleProto.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.ResourceHandleProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                        if (!(message.dtypes_and_shapes && message.dtypes_and_shapes.length))
                            message.dtypes_and_shapes = [];
                        message.dtypes_and_shapes.push($root.tensorflow.ResourceHandleProto.DtypeAndShape.decodeText(reader, true));
                        break;
                    case "allowed_devices":
                        if (!(message.allowed_devices && message.allowed_devices.length))
                            message.allowed_devices = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.allowed_devices.push(reader.string());
                                reader.next();
                            }
                        else
                            message.allowed_devices.push(reader.string());
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ResourceHandleProto.DtypeAndShape = (function() {
    
                function DtypeAndShape(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                DtypeAndShape.prototype.dtype = 0;
                DtypeAndShape.prototype.shape = null;
    
                DtypeAndShape.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.ResourceHandleProto.DtypeAndShape();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                DtypeAndShape.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.ResourceHandleProto.DtypeAndShape();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
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
    
        tensorflow.SavedObjectGraph = (function() {
    
            function SavedObjectGraph(properties) {
                this.nodes = [];
                this.concrete_functions = {};
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedObjectGraph.prototype.nodes = $util.emptyArray;
            SavedObjectGraph.prototype.concrete_functions = $util.emptyObject;
    
            SavedObjectGraph.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedObjectGraph(), key;
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.nodes && message.nodes.length))
                            message.nodes = [];
                        message.nodes.push($root.tensorflow.SavedObject.decode(reader, reader.uint32()));
                        break;
                    case 2:
                        reader.skip().pos++;
                        if (message.concrete_functions === $util.emptyObject)
                            message.concrete_functions = {};
                        key = reader.string();
                        reader.pos++;
                        message.concrete_functions[key] = $root.tensorflow.SavedConcreteFunction.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SavedObjectGraph.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SavedObjectGraph(), key, value;
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "nodes":
                        if (!(message.nodes && message.nodes.length))
                            message.nodes = [];
                        message.nodes.push($root.tensorflow.SavedObject.decodeText(reader, true));
                        break;
                    case "concrete_functions":
                        if (message.concrete_functions === $util.emptyObject)
                            message.concrete_functions = {};
                        reader.start();
                        key = "";
                        value = null;
                        while (!reader.end())
                            switch (reader.tag()) {
                            case "key":
                                key = reader.string();
                                break;
                            case "value":
                                value = $root.tensorflow.SavedConcreteFunction.decodeText(reader, true);
                                break;
                            }
                        message.concrete_functions[key] = value;
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
    
            function SavedObject(properties) {
                this.children = [];
                this.slot_variables = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedObject.prototype.children = $util.emptyArray;
            SavedObject.prototype.slot_variables = $util.emptyArray;
            SavedObject.prototype.user_object = null;
            SavedObject.prototype.asset = null;
            SavedObject.prototype["function"] = null;
            SavedObject.prototype.variable = null;
            SavedObject.prototype.bare_concrete_function = null;
            SavedObject.prototype.constant = null;
            SavedObject.prototype.resource = null;
    
            var $oneOfFields;
    
            Object.defineProperty(SavedObject.prototype, "kind", {
                get: $util.oneOfGetter($oneOfFields = ["user_object", "asset", "function", "variable", "bare_concrete_function", "constant", "resource"]),
                set: $util.oneOfSetter($oneOfFields)
            });
    
            SavedObject.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedObject();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.children && message.children.length))
                            message.children = [];
                        message.children.push($root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decode(reader, reader.uint32()));
                        break;
                    case 3:
                        if (!(message.slot_variables && message.slot_variables.length))
                            message.slot_variables = [];
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
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SavedObject.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SavedObject();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "children":
                        if (!(message.children && message.children.length))
                            message.children = [];
                        message.children.push($root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decodeText(reader, true));
                        break;
                    case "slot_variables":
                        if (!(message.slot_variables && message.slot_variables.length))
                            message.slot_variables = [];
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
    
            function SavedUserObject(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedUserObject.prototype.identifier = "";
            SavedUserObject.prototype.version = null;
            SavedUserObject.prototype.metadata = "";
    
            SavedUserObject.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedUserObject();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            SavedUserObject.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SavedUserObject();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function SavedAsset(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedAsset.prototype.asset_file_def_index = 0;
    
            SavedAsset.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedAsset();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            SavedAsset.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SavedAsset();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function SavedFunction(properties) {
                this.concrete_functions = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedFunction.prototype.concrete_functions = $util.emptyArray;
            SavedFunction.prototype.function_spec = null;
    
            SavedFunction.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedFunction();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.concrete_functions && message.concrete_functions.length))
                            message.concrete_functions = [];
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
    
            SavedFunction.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SavedFunction();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "concrete_functions":
                        if (!(message.concrete_functions && message.concrete_functions.length))
                            message.concrete_functions = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.concrete_functions.push(reader.string());
                                reader.next();
                            }
                        else
                            message.concrete_functions.push(reader.string());
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
    
            function SavedConcreteFunction(properties) {
                this.bound_inputs = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedConcreteFunction.prototype.bound_inputs = $util.emptyArray;
            SavedConcreteFunction.prototype.canonicalized_input_signature = null;
            SavedConcreteFunction.prototype.output_signature = null;
    
            SavedConcreteFunction.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedConcreteFunction();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 2:
                        if (!(message.bound_inputs && message.bound_inputs.length))
                            message.bound_inputs = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.bound_inputs.push(reader.int32());
                        } else
                            message.bound_inputs.push(reader.int32());
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
    
            SavedConcreteFunction.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SavedConcreteFunction();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "bound_inputs":
                        if (!(message.bound_inputs && message.bound_inputs.length))
                            message.bound_inputs = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.bound_inputs.push(reader.int32());
                                reader.next();
                            }
                        else
                            message.bound_inputs.push(reader.int32());
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
    
            function SavedBareConcreteFunction(properties) {
                this.argument_keywords = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedBareConcreteFunction.prototype.concrete_function_name = "";
            SavedBareConcreteFunction.prototype.argument_keywords = $util.emptyArray;
            SavedBareConcreteFunction.prototype.allowed_positional_arguments = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
            SavedBareConcreteFunction.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedBareConcreteFunction();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.concrete_function_name = reader.string();
                        break;
                    case 2:
                        if (!(message.argument_keywords && message.argument_keywords.length))
                            message.argument_keywords = [];
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
    
            SavedBareConcreteFunction.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SavedBareConcreteFunction();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "concrete_function_name":
                        message.concrete_function_name = reader.string();
                        break;
                    case "argument_keywords":
                        if (!(message.argument_keywords && message.argument_keywords.length))
                            message.argument_keywords = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.argument_keywords.push(reader.string());
                                reader.next();
                            }
                        else
                            message.argument_keywords.push(reader.string());
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
    
            function SavedConstant(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedConstant.prototype.operation = "";
    
            SavedConstant.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedConstant();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            SavedConstant.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SavedConstant();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function SavedVariable(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedVariable.prototype.dtype = 0;
            SavedVariable.prototype.shape = null;
            SavedVariable.prototype.trainable = false;
            SavedVariable.prototype.synchronization = 0;
            SavedVariable.prototype.aggregation = 0;
            SavedVariable.prototype.name = "";
    
            SavedVariable.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedVariable();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            SavedVariable.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SavedVariable();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function FunctionSpec(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            FunctionSpec.prototype.fullargspec = null;
            FunctionSpec.prototype.is_method = false;
            FunctionSpec.prototype.input_signature = null;
    
            FunctionSpec.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.FunctionSpec();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            FunctionSpec.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.FunctionSpec();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function SavedResource(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedResource.prototype.device = "";
    
            SavedResource.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedResource();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            SavedResource.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SavedResource();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
        tensorflow.TrackableObjectGraph = (function() {
    
            function TrackableObjectGraph(properties) {
                this.nodes = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TrackableObjectGraph.prototype.nodes = $util.emptyArray;
    
            TrackableObjectGraph.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TrackableObjectGraph();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.nodes && message.nodes.length))
                            message.nodes = [];
                        message.nodes.push($root.tensorflow.TrackableObjectGraph.TrackableObject.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TrackableObjectGraph.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.TrackableObjectGraph();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "nodes":
                        if (!(message.nodes && message.nodes.length))
                            message.nodes = [];
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
    
                function TrackableObject(properties) {
                    this.children = [];
                    this.attributes = [];
                    this.slot_variables = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                TrackableObject.prototype.children = $util.emptyArray;
                TrackableObject.prototype.attributes = $util.emptyArray;
                TrackableObject.prototype.slot_variables = $util.emptyArray;
    
                TrackableObject.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TrackableObjectGraph.TrackableObject();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.children && message.children.length))
                                message.children = [];
                            message.children.push($root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decode(reader, reader.uint32()));
                            break;
                        case 2:
                            if (!(message.attributes && message.attributes.length))
                                message.attributes = [];
                            message.attributes.push($root.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.decode(reader, reader.uint32()));
                            break;
                        case 3:
                            if (!(message.slot_variables && message.slot_variables.length))
                                message.slot_variables = [];
                            message.slot_variables.push($root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                TrackableObject.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.TrackableObjectGraph.TrackableObject();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "children":
                            if (!(message.children && message.children.length))
                                message.children = [];
                            message.children.push($root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decodeText(reader, true));
                            break;
                        case "attributes":
                            if (!(message.attributes && message.attributes.length))
                                message.attributes = [];
                            message.attributes.push($root.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.decodeText(reader, true));
                            break;
                        case "slot_variables":
                            if (!(message.slot_variables && message.slot_variables.length))
                                message.slot_variables = [];
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
    
                    function ObjectReference(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    ObjectReference.prototype.node_id = 0;
                    ObjectReference.prototype.local_name = "";
    
                    ObjectReference.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                    ObjectReference.decodeText = function decodeText(reader) {
                        var message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference();
                        reader.start();
                        while (!reader.end()) {
                            var tag = reader.tag();
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
    
                    function SerializedTensor(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    SerializedTensor.prototype.name = "";
                    SerializedTensor.prototype.full_name = "";
                    SerializedTensor.prototype.checkpoint_key = "";
                    SerializedTensor.prototype.optional_restore = false;
    
                    SerializedTensor.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                    SerializedTensor.decodeText = function decodeText(reader) {
                        var message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor();
                        reader.start();
                        while (!reader.end()) {
                            var tag = reader.tag();
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
    
                    function SlotVariableReference(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    SlotVariableReference.prototype.original_variable_node_id = 0;
                    SlotVariableReference.prototype.slot_name = "";
                    SlotVariableReference.prototype.slot_variable_node_id = 0;
    
                    SlotVariableReference.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                    SlotVariableReference.decodeText = function decodeText(reader) {
                        var message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference();
                        reader.start();
                        while (!reader.end()) {
                            var tag = reader.tag();
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
    
        tensorflow.StructuredValue = (function() {
    
            function StructuredValue(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            StructuredValue.prototype.none_value = null;
            StructuredValue.prototype.float64_value = 0;
            StructuredValue.prototype.int64_value = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
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
    
            var $oneOfFields;
    
            Object.defineProperty(StructuredValue.prototype, "kind", {
                get: $util.oneOfGetter($oneOfFields = ["none_value", "float64_value", "int64_value", "string_value", "bool_value", "tensor_shape_value", "tensor_dtype_value", "tensor_spec_value", "type_spec_value", "bounded_tensor_spec_value", "list_value", "tuple_value", "dict_value", "named_tuple_value"]),
                set: $util.oneOfSetter($oneOfFields)
            });
    
            StructuredValue.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.StructuredValue();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            StructuredValue.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.StructuredValue();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function NoneValue(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            NoneValue.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.NoneValue();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            NoneValue.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.NoneValue();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function ListValue(properties) {
                this.values = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ListValue.prototype.values = $util.emptyArray;
    
            ListValue.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.ListValue();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.values && message.values.length))
                            message.values = [];
                        message.values.push($root.tensorflow.StructuredValue.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ListValue.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.ListValue();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "values":
                        if (!(message.values && message.values.length))
                            message.values = [];
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
    
            function TupleValue(properties) {
                this.values = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TupleValue.prototype.values = $util.emptyArray;
    
            TupleValue.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TupleValue();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.values && message.values.length))
                            message.values = [];
                        message.values.push($root.tensorflow.StructuredValue.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TupleValue.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.TupleValue();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "values":
                        if (!(message.values && message.values.length))
                            message.values = [];
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
    
            function DictValue(properties) {
                this.fields = {};
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            DictValue.prototype.fields = $util.emptyObject;
    
            DictValue.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.DictValue(), key;
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        reader.skip().pos++;
                        if (message.fields === $util.emptyObject)
                            message.fields = {};
                        key = reader.string();
                        reader.pos++;
                        message.fields[key] = $root.tensorflow.StructuredValue.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            DictValue.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.DictValue(), key, value;
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "fields":
                        if (message.fields === $util.emptyObject)
                            message.fields = {};
                        reader.start();
                        key = "";
                        value = null;
                        while (!reader.end())
                            switch (reader.tag()) {
                            case "key":
                                key = reader.string();
                                break;
                            case "value":
                                value = $root.tensorflow.StructuredValue.decodeText(reader, true);
                                break;
                            }
                        message.fields[key] = value;
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
    
            function PairValue(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            PairValue.prototype.key = "";
            PairValue.prototype.value = null;
    
            PairValue.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.PairValue();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            PairValue.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.PairValue();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function NamedTupleValue(properties) {
                this.values = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            NamedTupleValue.prototype.name = "";
            NamedTupleValue.prototype.values = $util.emptyArray;
    
            NamedTupleValue.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.NamedTupleValue();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        if (!(message.values && message.values.length))
                            message.values = [];
                        message.values.push($root.tensorflow.PairValue.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            NamedTupleValue.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.NamedTupleValue();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "values":
                        if (!(message.values && message.values.length))
                            message.values = [];
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
    
            function TensorSpecProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorSpecProto.prototype.name = "";
            TensorSpecProto.prototype.shape = null;
            TensorSpecProto.prototype.dtype = 0;
    
            TensorSpecProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TensorSpecProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            TensorSpecProto.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.TensorSpecProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function BoundedTensorSpecProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            BoundedTensorSpecProto.prototype.name = "";
            BoundedTensorSpecProto.prototype.shape = null;
            BoundedTensorSpecProto.prototype.dtype = 0;
            BoundedTensorSpecProto.prototype.minimum = null;
            BoundedTensorSpecProto.prototype.maximum = null;
    
            BoundedTensorSpecProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.BoundedTensorSpecProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            BoundedTensorSpecProto.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.BoundedTensorSpecProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function TypeSpecProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TypeSpecProto.prototype.type_spec_class = 0;
            TypeSpecProto.prototype.type_state = null;
            TypeSpecProto.prototype.type_spec_class_name = "";
    
            TypeSpecProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TypeSpecProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            TypeSpecProto.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.TypeSpecProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "UNKNOWN"] = 0;
                values[valuesById[1] = "SPARSE_TENSOR_SPEC"] = 1;
                values[valuesById[2] = "INDEXED_SLICES_SPEC"] = 2;
                values[valuesById[3] = "RAGGED_TENSOR_SPEC"] = 3;
                values[valuesById[4] = "TENSOR_ARRAY_SPEC"] = 4;
                values[valuesById[5] = "DATA_DATASET_SPEC"] = 5;
                values[valuesById[6] = "DATA_ITERATOR_SPEC"] = 6;
                values[valuesById[7] = "OPTIONAL_SPEC"] = 7;
                values[valuesById[8] = "PER_REPLICA_SPEC"] = 8;
                values[valuesById[9] = "VARIABLE_SPEC"] = 9;
                values[valuesById[10] = "ROW_PARTITION_SPEC"] = 10;
                return values;
            })();
    
            return TypeSpecProto;
        })();
    
        tensorflow.BundleHeaderProto = (function() {
    
            function BundleHeaderProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            BundleHeaderProto.prototype.num_shards = 0;
            BundleHeaderProto.prototype.endianness = 0;
            BundleHeaderProto.prototype.version = null;
    
            BundleHeaderProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.BundleHeaderProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            BundleHeaderProto.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.BundleHeaderProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "LITTLE"] = 0;
                values[valuesById[1] = "BIG"] = 1;
                return values;
            })();
    
            return BundleHeaderProto;
        })();
    
        tensorflow.BundleEntryProto = (function() {
    
            function BundleEntryProto(properties) {
                this.slices = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            BundleEntryProto.prototype.dtype = 0;
            BundleEntryProto.prototype.shape = null;
            BundleEntryProto.prototype.shard_id = 0;
            BundleEntryProto.prototype.offset = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            BundleEntryProto.prototype.size = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            BundleEntryProto.prototype.crc32c = 0;
            BundleEntryProto.prototype.slices = $util.emptyArray;
    
            BundleEntryProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.BundleEntryProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
                        if (!(message.slices && message.slices.length))
                            message.slices = [];
                        message.slices.push($root.tensorflow.TensorSliceProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            BundleEntryProto.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.BundleEntryProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                        if (!(message.slices && message.slices.length))
                            message.slices = [];
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
    
            function TensorSliceProto(properties) {
                this.extent = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorSliceProto.prototype.extent = $util.emptyArray;
    
            TensorSliceProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TensorSliceProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.extent && message.extent.length))
                            message.extent = [];
                        message.extent.push($root.tensorflow.TensorSliceProto.Extent.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TensorSliceProto.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.TensorSliceProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "extent":
                        if (!(message.extent && message.extent.length))
                            message.extent = [];
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
    
                function Extent(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Extent.prototype.start = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                Extent.prototype.length = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                var $oneOfFields;
    
                Object.defineProperty(Extent.prototype, "has_length", {
                    get: $util.oneOfGetter($oneOfFields = ["length"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                Extent.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TensorSliceProto.Extent();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                Extent.decodeText = function decodeText(reader) {
                    var message = new $root.tensorflow.TensorSliceProto.Extent();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
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
    
            function SavedSliceMeta(properties) {
                this.slice = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedSliceMeta.prototype.name = "";
            SavedSliceMeta.prototype.shape = null;
            SavedSliceMeta.prototype.type = 0;
            SavedSliceMeta.prototype.slice = $util.emptyArray;
    
            SavedSliceMeta.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedSliceMeta();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
                        if (!(message.slice && message.slice.length))
                            message.slice = [];
                        message.slice.push($root.tensorflow.TensorSliceProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SavedSliceMeta.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SavedSliceMeta();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                        if (!(message.slice && message.slice.length))
                            message.slice = [];
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
    
            function SavedTensorSliceMeta(properties) {
                this.tensor = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedTensorSliceMeta.prototype.tensor = $util.emptyArray;
            SavedTensorSliceMeta.prototype.versions = null;
    
            SavedTensorSliceMeta.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedTensorSliceMeta();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.tensor && message.tensor.length))
                            message.tensor = [];
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
    
            SavedTensorSliceMeta.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SavedTensorSliceMeta();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "tensor":
                        if (!(message.tensor && message.tensor.length))
                            message.tensor = [];
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
    
            function SavedSlice(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedSlice.prototype.name = "";
            SavedSlice.prototype.slice = null;
            SavedSlice.prototype.data = null;
    
            SavedSlice.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedSlice();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            SavedSlice.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SavedSlice();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function SavedTensorSlices(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedTensorSlices.prototype.meta = null;
            SavedTensorSlices.prototype.data = null;
    
            SavedTensorSlices.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedTensorSlices();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            SavedTensorSlices.decodeText = function decodeText(reader) {
                var message = new $root.tensorflow.SavedTensorSlices();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
        var google = {};
    
        google.protobuf = (function() {
    
            var protobuf = {};
    
            protobuf.Any = (function() {
    
                function Any(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Any.prototype.type_url = "";
                Any.prototype.value = $util.newBuffer([]);
    
                Any.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.google.protobuf.Any();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                Any.decodeText = function decodeText(reader) {
                    var message = new $root.google.protobuf.Any();
                    reader.start();
                    if (reader.any(message))
                        return message;
                    while (!reader.end()) {
                        var tag = reader.tag();
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
