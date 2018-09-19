/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $util = $protobuf.util;
    
    var $root = $protobuf.roots.tf || ($protobuf.roots.tf = {});
    
    $root.tensorflow = (function() {
    
        var tensorflow = {};
    
        tensorflow.SavedModel = (function() {
    
            function SavedModel(properties) {
                this.metaGraphs = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SavedModel.prototype.savedModelSchemaVersion = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            SavedModel.prototype.metaGraphs = $util.emptyArray;
    
            SavedModel.create = function create(properties) {
                return new SavedModel(properties);
            };
    
            SavedModel.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SavedModel();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.savedModelSchemaVersion = reader.int64();
                        break;
                    case 2:
                        if (!(message.metaGraphs && message.metaGraphs.length))
                            message.metaGraphs = [];
                        message.metaGraphs.push($root.tensorflow.MetaGraphDef.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SavedModel.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.savedModelSchemaVersion != null && message.hasOwnProperty("savedModelSchemaVersion"))
                    if (!$util.isInteger(message.savedModelSchemaVersion) && !(message.savedModelSchemaVersion && $util.isInteger(message.savedModelSchemaVersion.low) && $util.isInteger(message.savedModelSchemaVersion.high)))
                        return "savedModelSchemaVersion: integer|Long expected";
                if (message.metaGraphs != null && message.hasOwnProperty("metaGraphs")) {
                    if (!Array.isArray(message.metaGraphs))
                        return "metaGraphs: array expected";
                    for (var i = 0; i < message.metaGraphs.length; ++i) {
                        var error = $root.tensorflow.MetaGraphDef.verify(message.metaGraphs[i]);
                        if (error)
                            return "metaGraphs." + error;
                    }
                }
                return null;
            };
    
            SavedModel.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.SavedModel)
                    return object;
                var message = new $root.tensorflow.SavedModel();
                if (object.savedModelSchemaVersion != null)
                    if ($util.Long)
                        (message.savedModelSchemaVersion = $util.Long.fromValue(object.savedModelSchemaVersion)).unsigned = false;
                    else if (typeof object.savedModelSchemaVersion === "string")
                        message.savedModelSchemaVersion = parseInt(object.savedModelSchemaVersion, 10);
                    else if (typeof object.savedModelSchemaVersion === "number")
                        message.savedModelSchemaVersion = object.savedModelSchemaVersion;
                    else if (typeof object.savedModelSchemaVersion === "object")
                        message.savedModelSchemaVersion = new $util.LongBits(object.savedModelSchemaVersion.low >>> 0, object.savedModelSchemaVersion.high >>> 0).toNumber();
                if (object.metaGraphs) {
                    if (!Array.isArray(object.metaGraphs))
                        throw TypeError(".tensorflow.SavedModel.metaGraphs: array expected");
                    message.metaGraphs = [];
                    for (var i = 0; i < object.metaGraphs.length; ++i) {
                        if (typeof object.metaGraphs[i] !== "object")
                            throw TypeError(".tensorflow.SavedModel.metaGraphs: object expected");
                        message.metaGraphs[i] = $root.tensorflow.MetaGraphDef.fromObject(object.metaGraphs[i]);
                    }
                }
                return message;
            };
    
            SavedModel.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.metaGraphs = [];
                if (options.defaults)
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.savedModelSchemaVersion = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.savedModelSchemaVersion = options.longs === String ? "0" : 0;
                if (message.savedModelSchemaVersion != null && message.hasOwnProperty("savedModelSchemaVersion"))
                    if (typeof message.savedModelSchemaVersion === "number")
                        object.savedModelSchemaVersion = options.longs === String ? String(message.savedModelSchemaVersion) : message.savedModelSchemaVersion;
                    else
                        object.savedModelSchemaVersion = options.longs === String ? $util.Long.prototype.toString.call(message.savedModelSchemaVersion) : options.longs === Number ? new $util.LongBits(message.savedModelSchemaVersion.low >>> 0, message.savedModelSchemaVersion.high >>> 0).toNumber() : message.savedModelSchemaVersion;
                if (message.metaGraphs && message.metaGraphs.length) {
                    object.metaGraphs = [];
                    for (var j = 0; j < message.metaGraphs.length; ++j)
                        object.metaGraphs[j] = $root.tensorflow.MetaGraphDef.toObject(message.metaGraphs[j], options);
                }
                return object;
            };
    
            SavedModel.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return SavedModel;
        })();
    
        tensorflow.MetaGraphDef = (function() {
    
            function MetaGraphDef(properties) {
                this.collectionDef = {};
                this.signatureDef = {};
                this.assetFileDef = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            MetaGraphDef.prototype.metaInfoDef = null;
            MetaGraphDef.prototype.graphDef = null;
            MetaGraphDef.prototype.saverDef = null;
            MetaGraphDef.prototype.collectionDef = $util.emptyObject;
            MetaGraphDef.prototype.signatureDef = $util.emptyObject;
            MetaGraphDef.prototype.assetFileDef = $util.emptyArray;
    
            MetaGraphDef.create = function create(properties) {
                return new MetaGraphDef(properties);
            };
    
            MetaGraphDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.MetaGraphDef(), key;
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.metaInfoDef = $root.tensorflow.MetaGraphDef.MetaInfoDef.decode(reader, reader.uint32());
                        break;
                    case 2:
                        message.graphDef = $root.tensorflow.GraphDef.decode(reader, reader.uint32());
                        break;
                    case 3:
                        message.saverDef = $root.tensorflow.SaverDef.decode(reader, reader.uint32());
                        break;
                    case 4:
                        reader.skip().pos++;
                        if (message.collectionDef === $util.emptyObject)
                            message.collectionDef = {};
                        key = reader.string();
                        reader.pos++;
                        message.collectionDef[key] = $root.tensorflow.CollectionDef.decode(reader, reader.uint32());
                        break;
                    case 5:
                        reader.skip().pos++;
                        if (message.signatureDef === $util.emptyObject)
                            message.signatureDef = {};
                        key = reader.string();
                        reader.pos++;
                        message.signatureDef[key] = $root.tensorflow.SignatureDef.decode(reader, reader.uint32());
                        break;
                    case 6:
                        if (!(message.assetFileDef && message.assetFileDef.length))
                            message.assetFileDef = [];
                        message.assetFileDef.push($root.tensorflow.AssetFileDef.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            MetaGraphDef.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.metaInfoDef != null && message.hasOwnProperty("metaInfoDef")) {
                    var error = $root.tensorflow.MetaGraphDef.MetaInfoDef.verify(message.metaInfoDef);
                    if (error)
                        return "metaInfoDef." + error;
                }
                if (message.graphDef != null && message.hasOwnProperty("graphDef")) {
                    var error = $root.tensorflow.GraphDef.verify(message.graphDef);
                    if (error)
                        return "graphDef." + error;
                }
                if (message.saverDef != null && message.hasOwnProperty("saverDef")) {
                    var error = $root.tensorflow.SaverDef.verify(message.saverDef);
                    if (error)
                        return "saverDef." + error;
                }
                if (message.collectionDef != null && message.hasOwnProperty("collectionDef")) {
                    if (!$util.isObject(message.collectionDef))
                        return "collectionDef: object expected";
                    var key = Object.keys(message.collectionDef);
                    for (var i = 0; i < key.length; ++i) {
                        var error = $root.tensorflow.CollectionDef.verify(message.collectionDef[key[i]]);
                        if (error)
                            return "collectionDef." + error;
                    }
                }
                if (message.signatureDef != null && message.hasOwnProperty("signatureDef")) {
                    if (!$util.isObject(message.signatureDef))
                        return "signatureDef: object expected";
                    var key = Object.keys(message.signatureDef);
                    for (var i = 0; i < key.length; ++i) {
                        var error = $root.tensorflow.SignatureDef.verify(message.signatureDef[key[i]]);
                        if (error)
                            return "signatureDef." + error;
                    }
                }
                if (message.assetFileDef != null && message.hasOwnProperty("assetFileDef")) {
                    if (!Array.isArray(message.assetFileDef))
                        return "assetFileDef: array expected";
                    for (var i = 0; i < message.assetFileDef.length; ++i) {
                        var error = $root.tensorflow.AssetFileDef.verify(message.assetFileDef[i]);
                        if (error)
                            return "assetFileDef." + error;
                    }
                }
                return null;
            };
    
            MetaGraphDef.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.MetaGraphDef)
                    return object;
                var message = new $root.tensorflow.MetaGraphDef();
                if (object.metaInfoDef != null) {
                    if (typeof object.metaInfoDef !== "object")
                        throw TypeError(".tensorflow.MetaGraphDef.metaInfoDef: object expected");
                    message.metaInfoDef = $root.tensorflow.MetaGraphDef.MetaInfoDef.fromObject(object.metaInfoDef);
                }
                if (object.graphDef != null) {
                    if (typeof object.graphDef !== "object")
                        throw TypeError(".tensorflow.MetaGraphDef.graphDef: object expected");
                    message.graphDef = $root.tensorflow.GraphDef.fromObject(object.graphDef);
                }
                if (object.saverDef != null) {
                    if (typeof object.saverDef !== "object")
                        throw TypeError(".tensorflow.MetaGraphDef.saverDef: object expected");
                    message.saverDef = $root.tensorflow.SaverDef.fromObject(object.saverDef);
                }
                if (object.collectionDef) {
                    if (typeof object.collectionDef !== "object")
                        throw TypeError(".tensorflow.MetaGraphDef.collectionDef: object expected");
                    message.collectionDef = {};
                    for (var keys = Object.keys(object.collectionDef), i = 0; i < keys.length; ++i) {
                        if (typeof object.collectionDef[keys[i]] !== "object")
                            throw TypeError(".tensorflow.MetaGraphDef.collectionDef: object expected");
                        message.collectionDef[keys[i]] = $root.tensorflow.CollectionDef.fromObject(object.collectionDef[keys[i]]);
                    }
                }
                if (object.signatureDef) {
                    if (typeof object.signatureDef !== "object")
                        throw TypeError(".tensorflow.MetaGraphDef.signatureDef: object expected");
                    message.signatureDef = {};
                    for (var keys = Object.keys(object.signatureDef), i = 0; i < keys.length; ++i) {
                        if (typeof object.signatureDef[keys[i]] !== "object")
                            throw TypeError(".tensorflow.MetaGraphDef.signatureDef: object expected");
                        message.signatureDef[keys[i]] = $root.tensorflow.SignatureDef.fromObject(object.signatureDef[keys[i]]);
                    }
                }
                if (object.assetFileDef) {
                    if (!Array.isArray(object.assetFileDef))
                        throw TypeError(".tensorflow.MetaGraphDef.assetFileDef: array expected");
                    message.assetFileDef = [];
                    for (var i = 0; i < object.assetFileDef.length; ++i) {
                        if (typeof object.assetFileDef[i] !== "object")
                            throw TypeError(".tensorflow.MetaGraphDef.assetFileDef: object expected");
                        message.assetFileDef[i] = $root.tensorflow.AssetFileDef.fromObject(object.assetFileDef[i]);
                    }
                }
                return message;
            };
    
            MetaGraphDef.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.assetFileDef = [];
                if (options.objects || options.defaults) {
                    object.collectionDef = {};
                    object.signatureDef = {};
                }
                if (options.defaults) {
                    object.metaInfoDef = null;
                    object.graphDef = null;
                    object.saverDef = null;
                }
                if (message.metaInfoDef != null && message.hasOwnProperty("metaInfoDef"))
                    object.metaInfoDef = $root.tensorflow.MetaGraphDef.MetaInfoDef.toObject(message.metaInfoDef, options);
                if (message.graphDef != null && message.hasOwnProperty("graphDef"))
                    object.graphDef = $root.tensorflow.GraphDef.toObject(message.graphDef, options);
                if (message.saverDef != null && message.hasOwnProperty("saverDef"))
                    object.saverDef = $root.tensorflow.SaverDef.toObject(message.saverDef, options);
                var keys2;
                if (message.collectionDef && (keys2 = Object.keys(message.collectionDef)).length) {
                    object.collectionDef = {};
                    for (var j = 0; j < keys2.length; ++j)
                        object.collectionDef[keys2[j]] = $root.tensorflow.CollectionDef.toObject(message.collectionDef[keys2[j]], options);
                }
                if (message.signatureDef && (keys2 = Object.keys(message.signatureDef)).length) {
                    object.signatureDef = {};
                    for (var j = 0; j < keys2.length; ++j)
                        object.signatureDef[keys2[j]] = $root.tensorflow.SignatureDef.toObject(message.signatureDef[keys2[j]], options);
                }
                if (message.assetFileDef && message.assetFileDef.length) {
                    object.assetFileDef = [];
                    for (var j = 0; j < message.assetFileDef.length; ++j)
                        object.assetFileDef[j] = $root.tensorflow.AssetFileDef.toObject(message.assetFileDef[j], options);
                }
                return object;
            };
    
            MetaGraphDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            MetaGraphDef.MetaInfoDef = (function() {
    
                function MetaInfoDef(properties) {
                    this.tags = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                MetaInfoDef.prototype.metaGraphVersion = "";
                MetaInfoDef.prototype.strippedOpList = null;
                MetaInfoDef.prototype.anyInfo = null;
                MetaInfoDef.prototype.tags = $util.emptyArray;
                MetaInfoDef.prototype.tensorflowVersion = "";
                MetaInfoDef.prototype.tensorflowGitVersion = "";
                MetaInfoDef.prototype.strippedDefaultAttrs = false;
    
                MetaInfoDef.create = function create(properties) {
                    return new MetaInfoDef(properties);
                };
    
                MetaInfoDef.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.MetaGraphDef.MetaInfoDef();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.metaGraphVersion = reader.string();
                            break;
                        case 2:
                            message.strippedOpList = $root.tensorflow.OpList.decode(reader, reader.uint32());
                            break;
                        case 3:
                            message.anyInfo = $root.google.protobuf.Any.decode(reader, reader.uint32());
                            break;
                        case 4:
                            if (!(message.tags && message.tags.length))
                                message.tags = [];
                            message.tags.push(reader.string());
                            break;
                        case 5:
                            message.tensorflowVersion = reader.string();
                            break;
                        case 6:
                            message.tensorflowGitVersion = reader.string();
                            break;
                        case 7:
                            message.strippedDefaultAttrs = reader.bool();
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                MetaInfoDef.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.metaGraphVersion != null && message.hasOwnProperty("metaGraphVersion"))
                        if (!$util.isString(message.metaGraphVersion))
                            return "metaGraphVersion: string expected";
                    if (message.strippedOpList != null && message.hasOwnProperty("strippedOpList")) {
                        var error = $root.tensorflow.OpList.verify(message.strippedOpList);
                        if (error)
                            return "strippedOpList." + error;
                    }
                    if (message.anyInfo != null && message.hasOwnProperty("anyInfo")) {
                        var error = $root.google.protobuf.Any.verify(message.anyInfo);
                        if (error)
                            return "anyInfo." + error;
                    }
                    if (message.tags != null && message.hasOwnProperty("tags")) {
                        if (!Array.isArray(message.tags))
                            return "tags: array expected";
                        for (var i = 0; i < message.tags.length; ++i)
                            if (!$util.isString(message.tags[i]))
                                return "tags: string[] expected";
                    }
                    if (message.tensorflowVersion != null && message.hasOwnProperty("tensorflowVersion"))
                        if (!$util.isString(message.tensorflowVersion))
                            return "tensorflowVersion: string expected";
                    if (message.tensorflowGitVersion != null && message.hasOwnProperty("tensorflowGitVersion"))
                        if (!$util.isString(message.tensorflowGitVersion))
                            return "tensorflowGitVersion: string expected";
                    if (message.strippedDefaultAttrs != null && message.hasOwnProperty("strippedDefaultAttrs"))
                        if (typeof message.strippedDefaultAttrs !== "boolean")
                            return "strippedDefaultAttrs: boolean expected";
                    return null;
                };
    
                MetaInfoDef.fromObject = function fromObject(object) {
                    if (object instanceof $root.tensorflow.MetaGraphDef.MetaInfoDef)
                        return object;
                    var message = new $root.tensorflow.MetaGraphDef.MetaInfoDef();
                    if (object.metaGraphVersion != null)
                        message.metaGraphVersion = String(object.metaGraphVersion);
                    if (object.strippedOpList != null) {
                        if (typeof object.strippedOpList !== "object")
                            throw TypeError(".tensorflow.MetaGraphDef.MetaInfoDef.strippedOpList: object expected");
                        message.strippedOpList = $root.tensorflow.OpList.fromObject(object.strippedOpList);
                    }
                    if (object.anyInfo != null) {
                        if (typeof object.anyInfo !== "object")
                            throw TypeError(".tensorflow.MetaGraphDef.MetaInfoDef.anyInfo: object expected");
                        message.anyInfo = $root.google.protobuf.Any.fromObject(object.anyInfo);
                    }
                    if (object.tags) {
                        if (!Array.isArray(object.tags))
                            throw TypeError(".tensorflow.MetaGraphDef.MetaInfoDef.tags: array expected");
                        message.tags = [];
                        for (var i = 0; i < object.tags.length; ++i)
                            message.tags[i] = String(object.tags[i]);
                    }
                    if (object.tensorflowVersion != null)
                        message.tensorflowVersion = String(object.tensorflowVersion);
                    if (object.tensorflowGitVersion != null)
                        message.tensorflowGitVersion = String(object.tensorflowGitVersion);
                    if (object.strippedDefaultAttrs != null)
                        message.strippedDefaultAttrs = Boolean(object.strippedDefaultAttrs);
                    return message;
                };
    
                MetaInfoDef.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.arrays || options.defaults)
                        object.tags = [];
                    if (options.defaults) {
                        object.metaGraphVersion = "";
                        object.strippedOpList = null;
                        object.anyInfo = null;
                        object.tensorflowVersion = "";
                        object.tensorflowGitVersion = "";
                        object.strippedDefaultAttrs = false;
                    }
                    if (message.metaGraphVersion != null && message.hasOwnProperty("metaGraphVersion"))
                        object.metaGraphVersion = message.metaGraphVersion;
                    if (message.strippedOpList != null && message.hasOwnProperty("strippedOpList"))
                        object.strippedOpList = $root.tensorflow.OpList.toObject(message.strippedOpList, options);
                    if (message.anyInfo != null && message.hasOwnProperty("anyInfo"))
                        object.anyInfo = $root.google.protobuf.Any.toObject(message.anyInfo, options);
                    if (message.tags && message.tags.length) {
                        object.tags = [];
                        for (var j = 0; j < message.tags.length; ++j)
                            object.tags[j] = message.tags[j];
                    }
                    if (message.tensorflowVersion != null && message.hasOwnProperty("tensorflowVersion"))
                        object.tensorflowVersion = message.tensorflowVersion;
                    if (message.tensorflowGitVersion != null && message.hasOwnProperty("tensorflowGitVersion"))
                        object.tensorflowGitVersion = message.tensorflowGitVersion;
                    if (message.strippedDefaultAttrs != null && message.hasOwnProperty("strippedDefaultAttrs"))
                        object.strippedDefaultAttrs = message.strippedDefaultAttrs;
                    return object;
                };
    
                MetaInfoDef.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
            CollectionDef.prototype.nodeList = null;
            CollectionDef.prototype.bytesList = null;
            CollectionDef.prototype.int64List = null;
            CollectionDef.prototype.floatList = null;
            CollectionDef.prototype.anyList = null;
    
            var $oneOfFields;
    
            Object.defineProperty(CollectionDef.prototype, "kind", {
                get: $util.oneOfGetter($oneOfFields = ["nodeList", "bytesList", "int64List", "floatList", "anyList"]),
                set: $util.oneOfSetter($oneOfFields)
            });
    
            CollectionDef.create = function create(properties) {
                return new CollectionDef(properties);
            };
    
            CollectionDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.CollectionDef();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.nodeList = $root.tensorflow.CollectionDef.NodeList.decode(reader, reader.uint32());
                        break;
                    case 2:
                        message.bytesList = $root.tensorflow.CollectionDef.BytesList.decode(reader, reader.uint32());
                        break;
                    case 3:
                        message.int64List = $root.tensorflow.CollectionDef.Int64List.decode(reader, reader.uint32());
                        break;
                    case 4:
                        message.floatList = $root.tensorflow.CollectionDef.FloatList.decode(reader, reader.uint32());
                        break;
                    case 5:
                        message.anyList = $root.tensorflow.CollectionDef.AnyList.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            CollectionDef.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                var properties = {};
                if (message.nodeList != null && message.hasOwnProperty("nodeList")) {
                    properties.kind = 1;
                    {
                        var error = $root.tensorflow.CollectionDef.NodeList.verify(message.nodeList);
                        if (error)
                            return "nodeList." + error;
                    }
                }
                if (message.bytesList != null && message.hasOwnProperty("bytesList")) {
                    if (properties.kind === 1)
                        return "kind: multiple values";
                    properties.kind = 1;
                    {
                        var error = $root.tensorflow.CollectionDef.BytesList.verify(message.bytesList);
                        if (error)
                            return "bytesList." + error;
                    }
                }
                if (message.int64List != null && message.hasOwnProperty("int64List")) {
                    if (properties.kind === 1)
                        return "kind: multiple values";
                    properties.kind = 1;
                    {
                        var error = $root.tensorflow.CollectionDef.Int64List.verify(message.int64List);
                        if (error)
                            return "int64List." + error;
                    }
                }
                if (message.floatList != null && message.hasOwnProperty("floatList")) {
                    if (properties.kind === 1)
                        return "kind: multiple values";
                    properties.kind = 1;
                    {
                        var error = $root.tensorflow.CollectionDef.FloatList.verify(message.floatList);
                        if (error)
                            return "floatList." + error;
                    }
                }
                if (message.anyList != null && message.hasOwnProperty("anyList")) {
                    if (properties.kind === 1)
                        return "kind: multiple values";
                    properties.kind = 1;
                    {
                        var error = $root.tensorflow.CollectionDef.AnyList.verify(message.anyList);
                        if (error)
                            return "anyList." + error;
                    }
                }
                return null;
            };
    
            CollectionDef.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.CollectionDef)
                    return object;
                var message = new $root.tensorflow.CollectionDef();
                if (object.nodeList != null) {
                    if (typeof object.nodeList !== "object")
                        throw TypeError(".tensorflow.CollectionDef.nodeList: object expected");
                    message.nodeList = $root.tensorflow.CollectionDef.NodeList.fromObject(object.nodeList);
                }
                if (object.bytesList != null) {
                    if (typeof object.bytesList !== "object")
                        throw TypeError(".tensorflow.CollectionDef.bytesList: object expected");
                    message.bytesList = $root.tensorflow.CollectionDef.BytesList.fromObject(object.bytesList);
                }
                if (object.int64List != null) {
                    if (typeof object.int64List !== "object")
                        throw TypeError(".tensorflow.CollectionDef.int64List: object expected");
                    message.int64List = $root.tensorflow.CollectionDef.Int64List.fromObject(object.int64List);
                }
                if (object.floatList != null) {
                    if (typeof object.floatList !== "object")
                        throw TypeError(".tensorflow.CollectionDef.floatList: object expected");
                    message.floatList = $root.tensorflow.CollectionDef.FloatList.fromObject(object.floatList);
                }
                if (object.anyList != null) {
                    if (typeof object.anyList !== "object")
                        throw TypeError(".tensorflow.CollectionDef.anyList: object expected");
                    message.anyList = $root.tensorflow.CollectionDef.AnyList.fromObject(object.anyList);
                }
                return message;
            };
    
            CollectionDef.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (message.nodeList != null && message.hasOwnProperty("nodeList")) {
                    object.nodeList = $root.tensorflow.CollectionDef.NodeList.toObject(message.nodeList, options);
                    if (options.oneofs)
                        object.kind = "nodeList";
                }
                if (message.bytesList != null && message.hasOwnProperty("bytesList")) {
                    object.bytesList = $root.tensorflow.CollectionDef.BytesList.toObject(message.bytesList, options);
                    if (options.oneofs)
                        object.kind = "bytesList";
                }
                if (message.int64List != null && message.hasOwnProperty("int64List")) {
                    object.int64List = $root.tensorflow.CollectionDef.Int64List.toObject(message.int64List, options);
                    if (options.oneofs)
                        object.kind = "int64List";
                }
                if (message.floatList != null && message.hasOwnProperty("floatList")) {
                    object.floatList = $root.tensorflow.CollectionDef.FloatList.toObject(message.floatList, options);
                    if (options.oneofs)
                        object.kind = "floatList";
                }
                if (message.anyList != null && message.hasOwnProperty("anyList")) {
                    object.anyList = $root.tensorflow.CollectionDef.AnyList.toObject(message.anyList, options);
                    if (options.oneofs)
                        object.kind = "anyList";
                }
                return object;
            };
    
            CollectionDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                NodeList.create = function create(properties) {
                    return new NodeList(properties);
                };
    
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
    
                NodeList.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.value != null && message.hasOwnProperty("value")) {
                        if (!Array.isArray(message.value))
                            return "value: array expected";
                        for (var i = 0; i < message.value.length; ++i)
                            if (!$util.isString(message.value[i]))
                                return "value: string[] expected";
                    }
                    return null;
                };
    
                NodeList.fromObject = function fromObject(object) {
                    if (object instanceof $root.tensorflow.CollectionDef.NodeList)
                        return object;
                    var message = new $root.tensorflow.CollectionDef.NodeList();
                    if (object.value) {
                        if (!Array.isArray(object.value))
                            throw TypeError(".tensorflow.CollectionDef.NodeList.value: array expected");
                        message.value = [];
                        for (var i = 0; i < object.value.length; ++i)
                            message.value[i] = String(object.value[i]);
                    }
                    return message;
                };
    
                NodeList.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.arrays || options.defaults)
                        object.value = [];
                    if (message.value && message.value.length) {
                        object.value = [];
                        for (var j = 0; j < message.value.length; ++j)
                            object.value[j] = message.value[j];
                    }
                    return object;
                };
    
                NodeList.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                BytesList.create = function create(properties) {
                    return new BytesList(properties);
                };
    
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
    
                BytesList.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.value != null && message.hasOwnProperty("value")) {
                        if (!Array.isArray(message.value))
                            return "value: array expected";
                        for (var i = 0; i < message.value.length; ++i)
                            if (!(message.value[i] && typeof message.value[i].length === "number" || $util.isString(message.value[i])))
                                return "value: buffer[] expected";
                    }
                    return null;
                };
    
                BytesList.fromObject = function fromObject(object) {
                    if (object instanceof $root.tensorflow.CollectionDef.BytesList)
                        return object;
                    var message = new $root.tensorflow.CollectionDef.BytesList();
                    if (object.value) {
                        if (!Array.isArray(object.value))
                            throw TypeError(".tensorflow.CollectionDef.BytesList.value: array expected");
                        message.value = [];
                        for (var i = 0; i < object.value.length; ++i)
                            if (typeof object.value[i] === "string")
                                $util.base64.decode(object.value[i], message.value[i] = $util.newBuffer($util.base64.length(object.value[i])), 0);
                            else if (object.value[i].length)
                                message.value[i] = object.value[i];
                    }
                    return message;
                };
    
                BytesList.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.arrays || options.defaults)
                        object.value = [];
                    if (message.value && message.value.length) {
                        object.value = [];
                        for (var j = 0; j < message.value.length; ++j)
                            object.value[j] = options.bytes === String ? $util.base64.encode(message.value[j], 0, message.value[j].length) : options.bytes === Array ? Array.prototype.slice.call(message.value[j]) : message.value[j];
                    }
                    return object;
                };
    
                BytesList.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                Int64List.create = function create(properties) {
                    return new Int64List(properties);
                };
    
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
    
                Int64List.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.value != null && message.hasOwnProperty("value")) {
                        if (!Array.isArray(message.value))
                            return "value: array expected";
                        for (var i = 0; i < message.value.length; ++i)
                            if (!$util.isInteger(message.value[i]) && !(message.value[i] && $util.isInteger(message.value[i].low) && $util.isInteger(message.value[i].high)))
                                return "value: integer|Long[] expected";
                    }
                    return null;
                };
    
                Int64List.fromObject = function fromObject(object) {
                    if (object instanceof $root.tensorflow.CollectionDef.Int64List)
                        return object;
                    var message = new $root.tensorflow.CollectionDef.Int64List();
                    if (object.value) {
                        if (!Array.isArray(object.value))
                            throw TypeError(".tensorflow.CollectionDef.Int64List.value: array expected");
                        message.value = [];
                        for (var i = 0; i < object.value.length; ++i)
                            if ($util.Long)
                                (message.value[i] = $util.Long.fromValue(object.value[i])).unsigned = false;
                            else if (typeof object.value[i] === "string")
                                message.value[i] = parseInt(object.value[i], 10);
                            else if (typeof object.value[i] === "number")
                                message.value[i] = object.value[i];
                            else if (typeof object.value[i] === "object")
                                message.value[i] = new $util.LongBits(object.value[i].low >>> 0, object.value[i].high >>> 0).toNumber();
                    }
                    return message;
                };
    
                Int64List.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.arrays || options.defaults)
                        object.value = [];
                    if (message.value && message.value.length) {
                        object.value = [];
                        for (var j = 0; j < message.value.length; ++j)
                            if (typeof message.value[j] === "number")
                                object.value[j] = options.longs === String ? String(message.value[j]) : message.value[j];
                            else
                                object.value[j] = options.longs === String ? $util.Long.prototype.toString.call(message.value[j]) : options.longs === Number ? new $util.LongBits(message.value[j].low >>> 0, message.value[j].high >>> 0).toNumber() : message.value[j];
                    }
                    return object;
                };
    
                Int64List.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                FloatList.create = function create(properties) {
                    return new FloatList(properties);
                };
    
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
    
                FloatList.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.value != null && message.hasOwnProperty("value")) {
                        if (!Array.isArray(message.value))
                            return "value: array expected";
                        for (var i = 0; i < message.value.length; ++i)
                            if (typeof message.value[i] !== "number")
                                return "value: number[] expected";
                    }
                    return null;
                };
    
                FloatList.fromObject = function fromObject(object) {
                    if (object instanceof $root.tensorflow.CollectionDef.FloatList)
                        return object;
                    var message = new $root.tensorflow.CollectionDef.FloatList();
                    if (object.value) {
                        if (!Array.isArray(object.value))
                            throw TypeError(".tensorflow.CollectionDef.FloatList.value: array expected");
                        message.value = [];
                        for (var i = 0; i < object.value.length; ++i)
                            message.value[i] = Number(object.value[i]);
                    }
                    return message;
                };
    
                FloatList.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.arrays || options.defaults)
                        object.value = [];
                    if (message.value && message.value.length) {
                        object.value = [];
                        for (var j = 0; j < message.value.length; ++j)
                            object.value[j] = options.json && !isFinite(message.value[j]) ? String(message.value[j]) : message.value[j];
                    }
                    return object;
                };
    
                FloatList.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                AnyList.create = function create(properties) {
                    return new AnyList(properties);
                };
    
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
    
                AnyList.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.value != null && message.hasOwnProperty("value")) {
                        if (!Array.isArray(message.value))
                            return "value: array expected";
                        for (var i = 0; i < message.value.length; ++i) {
                            var error = $root.google.protobuf.Any.verify(message.value[i]);
                            if (error)
                                return "value." + error;
                        }
                    }
                    return null;
                };
    
                AnyList.fromObject = function fromObject(object) {
                    if (object instanceof $root.tensorflow.CollectionDef.AnyList)
                        return object;
                    var message = new $root.tensorflow.CollectionDef.AnyList();
                    if (object.value) {
                        if (!Array.isArray(object.value))
                            throw TypeError(".tensorflow.CollectionDef.AnyList.value: array expected");
                        message.value = [];
                        for (var i = 0; i < object.value.length; ++i) {
                            if (typeof object.value[i] !== "object")
                                throw TypeError(".tensorflow.CollectionDef.AnyList.value: object expected");
                            message.value[i] = $root.google.protobuf.Any.fromObject(object.value[i]);
                        }
                    }
                    return message;
                };
    
                AnyList.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.arrays || options.defaults)
                        object.value = [];
                    if (message.value && message.value.length) {
                        object.value = [];
                        for (var j = 0; j < message.value.length; ++j)
                            object.value[j] = $root.google.protobuf.Any.toObject(message.value[j], options);
                    }
                    return object;
                };
    
                AnyList.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
            TensorInfo.prototype.cooSparse = null;
            TensorInfo.prototype.dtype = 0;
            TensorInfo.prototype.tensorShape = null;
    
            var $oneOfFields;
    
            Object.defineProperty(TensorInfo.prototype, "encoding", {
                get: $util.oneOfGetter($oneOfFields = ["name", "cooSparse"]),
                set: $util.oneOfSetter($oneOfFields)
            });
    
            TensorInfo.create = function create(properties) {
                return new TensorInfo(properties);
            };
    
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
                        message.cooSparse = $root.tensorflow.TensorInfo.CooSparse.decode(reader, reader.uint32());
                        break;
                    case 2:
                        message.dtype = reader.int32();
                        break;
                    case 3:
                        message.tensorShape = $root.tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TensorInfo.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                var properties = {};
                if (message.name != null && message.hasOwnProperty("name")) {
                    properties.encoding = 1;
                    if (!$util.isString(message.name))
                        return "name: string expected";
                }
                if (message.cooSparse != null && message.hasOwnProperty("cooSparse")) {
                    if (properties.encoding === 1)
                        return "encoding: multiple values";
                    properties.encoding = 1;
                    {
                        var error = $root.tensorflow.TensorInfo.CooSparse.verify(message.cooSparse);
                        if (error)
                            return "cooSparse." + error;
                    }
                }
                if (message.dtype != null && message.hasOwnProperty("dtype"))
                    switch (message.dtype) {
                    default:
                        return "dtype: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                    case 3:
                    case 4:
                    case 5:
                    case 6:
                    case 7:
                    case 8:
                    case 9:
                    case 10:
                    case 11:
                    case 12:
                    case 13:
                    case 14:
                    case 15:
                    case 16:
                    case 17:
                    case 18:
                    case 19:
                    case 20:
                    case 21:
                    case 22:
                    case 23:
                    case 101:
                    case 102:
                    case 103:
                    case 104:
                    case 105:
                    case 106:
                    case 107:
                    case 108:
                    case 109:
                    case 110:
                    case 111:
                    case 112:
                    case 113:
                    case 114:
                    case 115:
                    case 116:
                    case 117:
                    case 118:
                    case 119:
                    case 120:
                    case 121:
                    case 122:
                    case 123:
                        break;
                    }
                if (message.tensorShape != null && message.hasOwnProperty("tensorShape")) {
                    var error = $root.tensorflow.TensorShapeProto.verify(message.tensorShape);
                    if (error)
                        return "tensorShape." + error;
                }
                return null;
            };
    
            TensorInfo.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.TensorInfo)
                    return object;
                var message = new $root.tensorflow.TensorInfo();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.cooSparse != null) {
                    if (typeof object.cooSparse !== "object")
                        throw TypeError(".tensorflow.TensorInfo.cooSparse: object expected");
                    message.cooSparse = $root.tensorflow.TensorInfo.CooSparse.fromObject(object.cooSparse);
                }
                switch (object.dtype) {
                case "DT_INVALID":
                case 0:
                    message.dtype = 0;
                    break;
                case "DT_FLOAT":
                case 1:
                    message.dtype = 1;
                    break;
                case "DT_DOUBLE":
                case 2:
                    message.dtype = 2;
                    break;
                case "DT_INT32":
                case 3:
                    message.dtype = 3;
                    break;
                case "DT_UINT8":
                case 4:
                    message.dtype = 4;
                    break;
                case "DT_INT16":
                case 5:
                    message.dtype = 5;
                    break;
                case "DT_INT8":
                case 6:
                    message.dtype = 6;
                    break;
                case "DT_STRING":
                case 7:
                    message.dtype = 7;
                    break;
                case "DT_COMPLEX64":
                case 8:
                    message.dtype = 8;
                    break;
                case "DT_INT64":
                case 9:
                    message.dtype = 9;
                    break;
                case "DT_BOOL":
                case 10:
                    message.dtype = 10;
                    break;
                case "DT_QINT8":
                case 11:
                    message.dtype = 11;
                    break;
                case "DT_QUINT8":
                case 12:
                    message.dtype = 12;
                    break;
                case "DT_QINT32":
                case 13:
                    message.dtype = 13;
                    break;
                case "DT_BFLOAT16":
                case 14:
                    message.dtype = 14;
                    break;
                case "DT_QINT16":
                case 15:
                    message.dtype = 15;
                    break;
                case "DT_QUINT16":
                case 16:
                    message.dtype = 16;
                    break;
                case "DT_UINT16":
                case 17:
                    message.dtype = 17;
                    break;
                case "DT_COMPLEX128":
                case 18:
                    message.dtype = 18;
                    break;
                case "DT_HALF":
                case 19:
                    message.dtype = 19;
                    break;
                case "DT_RESOURCE":
                case 20:
                    message.dtype = 20;
                    break;
                case "DT_VARIANT":
                case 21:
                    message.dtype = 21;
                    break;
                case "DT_UINT32":
                case 22:
                    message.dtype = 22;
                    break;
                case "DT_UINT64":
                case 23:
                    message.dtype = 23;
                    break;
                case "DT_FLOAT_REF":
                case 101:
                    message.dtype = 101;
                    break;
                case "DT_DOUBLE_REF":
                case 102:
                    message.dtype = 102;
                    break;
                case "DT_INT32_REF":
                case 103:
                    message.dtype = 103;
                    break;
                case "DT_UINT8_REF":
                case 104:
                    message.dtype = 104;
                    break;
                case "DT_INT16_REF":
                case 105:
                    message.dtype = 105;
                    break;
                case "DT_INT8_REF":
                case 106:
                    message.dtype = 106;
                    break;
                case "DT_STRING_REF":
                case 107:
                    message.dtype = 107;
                    break;
                case "DT_COMPLEX64_REF":
                case 108:
                    message.dtype = 108;
                    break;
                case "DT_INT64_REF":
                case 109:
                    message.dtype = 109;
                    break;
                case "DT_BOOL_REF":
                case 110:
                    message.dtype = 110;
                    break;
                case "DT_QINT8_REF":
                case 111:
                    message.dtype = 111;
                    break;
                case "DT_QUINT8_REF":
                case 112:
                    message.dtype = 112;
                    break;
                case "DT_QINT32_REF":
                case 113:
                    message.dtype = 113;
                    break;
                case "DT_BFLOAT16_REF":
                case 114:
                    message.dtype = 114;
                    break;
                case "DT_QINT16_REF":
                case 115:
                    message.dtype = 115;
                    break;
                case "DT_QUINT16_REF":
                case 116:
                    message.dtype = 116;
                    break;
                case "DT_UINT16_REF":
                case 117:
                    message.dtype = 117;
                    break;
                case "DT_COMPLEX128_REF":
                case 118:
                    message.dtype = 118;
                    break;
                case "DT_HALF_REF":
                case 119:
                    message.dtype = 119;
                    break;
                case "DT_RESOURCE_REF":
                case 120:
                    message.dtype = 120;
                    break;
                case "DT_VARIANT_REF":
                case 121:
                    message.dtype = 121;
                    break;
                case "DT_UINT32_REF":
                case 122:
                    message.dtype = 122;
                    break;
                case "DT_UINT64_REF":
                case 123:
                    message.dtype = 123;
                    break;
                }
                if (object.tensorShape != null) {
                    if (typeof object.tensorShape !== "object")
                        throw TypeError(".tensorflow.TensorInfo.tensorShape: object expected");
                    message.tensorShape = $root.tensorflow.TensorShapeProto.fromObject(object.tensorShape);
                }
                return message;
            };
    
            TensorInfo.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.dtype = options.enums === String ? "DT_INVALID" : 0;
                    object.tensorShape = null;
                }
                if (message.name != null && message.hasOwnProperty("name")) {
                    object.name = message.name;
                    if (options.oneofs)
                        object.encoding = "name";
                }
                if (message.dtype != null && message.hasOwnProperty("dtype"))
                    object.dtype = options.enums === String ? $root.tensorflow.DataType[message.dtype] : message.dtype;
                if (message.tensorShape != null && message.hasOwnProperty("tensorShape"))
                    object.tensorShape = $root.tensorflow.TensorShapeProto.toObject(message.tensorShape, options);
                if (message.cooSparse != null && message.hasOwnProperty("cooSparse")) {
                    object.cooSparse = $root.tensorflow.TensorInfo.CooSparse.toObject(message.cooSparse, options);
                    if (options.oneofs)
                        object.encoding = "cooSparse";
                }
                return object;
            };
    
            TensorInfo.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            TensorInfo.CooSparse = (function() {
    
                function CooSparse(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                CooSparse.prototype.valuesTensorName = "";
                CooSparse.prototype.indicesTensorName = "";
                CooSparse.prototype.denseShapeTensorName = "";
    
                CooSparse.create = function create(properties) {
                    return new CooSparse(properties);
                };
    
                CooSparse.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.TensorInfo.CooSparse();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.valuesTensorName = reader.string();
                            break;
                        case 2:
                            message.indicesTensorName = reader.string();
                            break;
                        case 3:
                            message.denseShapeTensorName = reader.string();
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                CooSparse.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.valuesTensorName != null && message.hasOwnProperty("valuesTensorName"))
                        if (!$util.isString(message.valuesTensorName))
                            return "valuesTensorName: string expected";
                    if (message.indicesTensorName != null && message.hasOwnProperty("indicesTensorName"))
                        if (!$util.isString(message.indicesTensorName))
                            return "indicesTensorName: string expected";
                    if (message.denseShapeTensorName != null && message.hasOwnProperty("denseShapeTensorName"))
                        if (!$util.isString(message.denseShapeTensorName))
                            return "denseShapeTensorName: string expected";
                    return null;
                };
    
                CooSparse.fromObject = function fromObject(object) {
                    if (object instanceof $root.tensorflow.TensorInfo.CooSparse)
                        return object;
                    var message = new $root.tensorflow.TensorInfo.CooSparse();
                    if (object.valuesTensorName != null)
                        message.valuesTensorName = String(object.valuesTensorName);
                    if (object.indicesTensorName != null)
                        message.indicesTensorName = String(object.indicesTensorName);
                    if (object.denseShapeTensorName != null)
                        message.denseShapeTensorName = String(object.denseShapeTensorName);
                    return message;
                };
    
                CooSparse.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        object.valuesTensorName = "";
                        object.indicesTensorName = "";
                        object.denseShapeTensorName = "";
                    }
                    if (message.valuesTensorName != null && message.hasOwnProperty("valuesTensorName"))
                        object.valuesTensorName = message.valuesTensorName;
                    if (message.indicesTensorName != null && message.hasOwnProperty("indicesTensorName"))
                        object.indicesTensorName = message.indicesTensorName;
                    if (message.denseShapeTensorName != null && message.hasOwnProperty("denseShapeTensorName"))
                        object.denseShapeTensorName = message.denseShapeTensorName;
                    return object;
                };
    
                CooSparse.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return CooSparse;
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
            SignatureDef.prototype.methodName = "";
    
            SignatureDef.create = function create(properties) {
                return new SignatureDef(properties);
            };
    
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
                        message.methodName = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SignatureDef.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.inputs != null && message.hasOwnProperty("inputs")) {
                    if (!$util.isObject(message.inputs))
                        return "inputs: object expected";
                    var key = Object.keys(message.inputs);
                    for (var i = 0; i < key.length; ++i) {
                        var error = $root.tensorflow.TensorInfo.verify(message.inputs[key[i]]);
                        if (error)
                            return "inputs." + error;
                    }
                }
                if (message.outputs != null && message.hasOwnProperty("outputs")) {
                    if (!$util.isObject(message.outputs))
                        return "outputs: object expected";
                    var key = Object.keys(message.outputs);
                    for (var i = 0; i < key.length; ++i) {
                        var error = $root.tensorflow.TensorInfo.verify(message.outputs[key[i]]);
                        if (error)
                            return "outputs." + error;
                    }
                }
                if (message.methodName != null && message.hasOwnProperty("methodName"))
                    if (!$util.isString(message.methodName))
                        return "methodName: string expected";
                return null;
            };
    
            SignatureDef.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.SignatureDef)
                    return object;
                var message = new $root.tensorflow.SignatureDef();
                if (object.inputs) {
                    if (typeof object.inputs !== "object")
                        throw TypeError(".tensorflow.SignatureDef.inputs: object expected");
                    message.inputs = {};
                    for (var keys = Object.keys(object.inputs), i = 0; i < keys.length; ++i) {
                        if (typeof object.inputs[keys[i]] !== "object")
                            throw TypeError(".tensorflow.SignatureDef.inputs: object expected");
                        message.inputs[keys[i]] = $root.tensorflow.TensorInfo.fromObject(object.inputs[keys[i]]);
                    }
                }
                if (object.outputs) {
                    if (typeof object.outputs !== "object")
                        throw TypeError(".tensorflow.SignatureDef.outputs: object expected");
                    message.outputs = {};
                    for (var keys = Object.keys(object.outputs), i = 0; i < keys.length; ++i) {
                        if (typeof object.outputs[keys[i]] !== "object")
                            throw TypeError(".tensorflow.SignatureDef.outputs: object expected");
                        message.outputs[keys[i]] = $root.tensorflow.TensorInfo.fromObject(object.outputs[keys[i]]);
                    }
                }
                if (object.methodName != null)
                    message.methodName = String(object.methodName);
                return message;
            };
    
            SignatureDef.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.objects || options.defaults) {
                    object.inputs = {};
                    object.outputs = {};
                }
                if (options.defaults)
                    object.methodName = "";
                var keys2;
                if (message.inputs && (keys2 = Object.keys(message.inputs)).length) {
                    object.inputs = {};
                    for (var j = 0; j < keys2.length; ++j)
                        object.inputs[keys2[j]] = $root.tensorflow.TensorInfo.toObject(message.inputs[keys2[j]], options);
                }
                if (message.outputs && (keys2 = Object.keys(message.outputs)).length) {
                    object.outputs = {};
                    for (var j = 0; j < keys2.length; ++j)
                        object.outputs[keys2[j]] = $root.tensorflow.TensorInfo.toObject(message.outputs[keys2[j]], options);
                }
                if (message.methodName != null && message.hasOwnProperty("methodName"))
                    object.methodName = message.methodName;
                return object;
            };
    
            SignatureDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
            AssetFileDef.prototype.tensorInfo = null;
            AssetFileDef.prototype.filename = "";
    
            AssetFileDef.create = function create(properties) {
                return new AssetFileDef(properties);
            };
    
            AssetFileDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.AssetFileDef();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.tensorInfo = $root.tensorflow.TensorInfo.decode(reader, reader.uint32());
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
    
            AssetFileDef.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.tensorInfo != null && message.hasOwnProperty("tensorInfo")) {
                    var error = $root.tensorflow.TensorInfo.verify(message.tensorInfo);
                    if (error)
                        return "tensorInfo." + error;
                }
                if (message.filename != null && message.hasOwnProperty("filename"))
                    if (!$util.isString(message.filename))
                        return "filename: string expected";
                return null;
            };
    
            AssetFileDef.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.AssetFileDef)
                    return object;
                var message = new $root.tensorflow.AssetFileDef();
                if (object.tensorInfo != null) {
                    if (typeof object.tensorInfo !== "object")
                        throw TypeError(".tensorflow.AssetFileDef.tensorInfo: object expected");
                    message.tensorInfo = $root.tensorflow.TensorInfo.fromObject(object.tensorInfo);
                }
                if (object.filename != null)
                    message.filename = String(object.filename);
                return message;
            };
    
            AssetFileDef.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.tensorInfo = null;
                    object.filename = "";
                }
                if (message.tensorInfo != null && message.hasOwnProperty("tensorInfo"))
                    object.tensorInfo = $root.tensorflow.TensorInfo.toObject(message.tensorInfo, options);
                if (message.filename != null && message.hasOwnProperty("filename"))
                    object.filename = message.filename;
                return object;
            };
    
            AssetFileDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
            SaverDef.prototype.filenameTensorName = "";
            SaverDef.prototype.saveTensorName = "";
            SaverDef.prototype.restoreOpName = "";
            SaverDef.prototype.maxToKeep = 0;
            SaverDef.prototype.sharded = false;
            SaverDef.prototype.keepCheckpointEveryNHours = 0;
            SaverDef.prototype.version = 0;
    
            SaverDef.create = function create(properties) {
                return new SaverDef(properties);
            };
    
            SaverDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.SaverDef();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.filenameTensorName = reader.string();
                        break;
                    case 2:
                        message.saveTensorName = reader.string();
                        break;
                    case 3:
                        message.restoreOpName = reader.string();
                        break;
                    case 4:
                        message.maxToKeep = reader.int32();
                        break;
                    case 5:
                        message.sharded = reader.bool();
                        break;
                    case 6:
                        message.keepCheckpointEveryNHours = reader.float();
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
    
            SaverDef.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.filenameTensorName != null && message.hasOwnProperty("filenameTensorName"))
                    if (!$util.isString(message.filenameTensorName))
                        return "filenameTensorName: string expected";
                if (message.saveTensorName != null && message.hasOwnProperty("saveTensorName"))
                    if (!$util.isString(message.saveTensorName))
                        return "saveTensorName: string expected";
                if (message.restoreOpName != null && message.hasOwnProperty("restoreOpName"))
                    if (!$util.isString(message.restoreOpName))
                        return "restoreOpName: string expected";
                if (message.maxToKeep != null && message.hasOwnProperty("maxToKeep"))
                    if (!$util.isInteger(message.maxToKeep))
                        return "maxToKeep: integer expected";
                if (message.sharded != null && message.hasOwnProperty("sharded"))
                    if (typeof message.sharded !== "boolean")
                        return "sharded: boolean expected";
                if (message.keepCheckpointEveryNHours != null && message.hasOwnProperty("keepCheckpointEveryNHours"))
                    if (typeof message.keepCheckpointEveryNHours !== "number")
                        return "keepCheckpointEveryNHours: number expected";
                if (message.version != null && message.hasOwnProperty("version"))
                    switch (message.version) {
                    default:
                        return "version: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                        break;
                    }
                return null;
            };
    
            SaverDef.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.SaverDef)
                    return object;
                var message = new $root.tensorflow.SaverDef();
                if (object.filenameTensorName != null)
                    message.filenameTensorName = String(object.filenameTensorName);
                if (object.saveTensorName != null)
                    message.saveTensorName = String(object.saveTensorName);
                if (object.restoreOpName != null)
                    message.restoreOpName = String(object.restoreOpName);
                if (object.maxToKeep != null)
                    message.maxToKeep = object.maxToKeep | 0;
                if (object.sharded != null)
                    message.sharded = Boolean(object.sharded);
                if (object.keepCheckpointEveryNHours != null)
                    message.keepCheckpointEveryNHours = Number(object.keepCheckpointEveryNHours);
                switch (object.version) {
                case "LEGACY":
                case 0:
                    message.version = 0;
                    break;
                case "V1":
                case 1:
                    message.version = 1;
                    break;
                case "V2":
                case 2:
                    message.version = 2;
                    break;
                }
                return message;
            };
    
            SaverDef.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.filenameTensorName = "";
                    object.saveTensorName = "";
                    object.restoreOpName = "";
                    object.maxToKeep = 0;
                    object.sharded = false;
                    object.keepCheckpointEveryNHours = 0;
                    object.version = options.enums === String ? "LEGACY" : 0;
                }
                if (message.filenameTensorName != null && message.hasOwnProperty("filenameTensorName"))
                    object.filenameTensorName = message.filenameTensorName;
                if (message.saveTensorName != null && message.hasOwnProperty("saveTensorName"))
                    object.saveTensorName = message.saveTensorName;
                if (message.restoreOpName != null && message.hasOwnProperty("restoreOpName"))
                    object.restoreOpName = message.restoreOpName;
                if (message.maxToKeep != null && message.hasOwnProperty("maxToKeep"))
                    object.maxToKeep = message.maxToKeep;
                if (message.sharded != null && message.hasOwnProperty("sharded"))
                    object.sharded = message.sharded;
                if (message.keepCheckpointEveryNHours != null && message.hasOwnProperty("keepCheckpointEveryNHours"))
                    object.keepCheckpointEveryNHours = options.json && !isFinite(message.keepCheckpointEveryNHours) ? String(message.keepCheckpointEveryNHours) : message.keepCheckpointEveryNHours;
                if (message.version != null && message.hasOwnProperty("version"))
                    object.version = options.enums === String ? $root.tensorflow.SaverDef.CheckpointFormatVersion[message.version] : message.version;
                return object;
            };
    
            SaverDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
            GraphDef.create = function create(properties) {
                return new GraphDef(properties);
            };
    
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
    
            GraphDef.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.node != null && message.hasOwnProperty("node")) {
                    if (!Array.isArray(message.node))
                        return "node: array expected";
                    for (var i = 0; i < message.node.length; ++i) {
                        var error = $root.tensorflow.NodeDef.verify(message.node[i]);
                        if (error)
                            return "node." + error;
                    }
                }
                if (message.versions != null && message.hasOwnProperty("versions")) {
                    var error = $root.tensorflow.VersionDef.verify(message.versions);
                    if (error)
                        return "versions." + error;
                }
                if (message.version != null && message.hasOwnProperty("version"))
                    if (!$util.isInteger(message.version))
                        return "version: integer expected";
                if (message.library != null && message.hasOwnProperty("library")) {
                    var error = $root.tensorflow.FunctionDefLibrary.verify(message.library);
                    if (error)
                        return "library." + error;
                }
                return null;
            };
    
            GraphDef.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.GraphDef)
                    return object;
                var message = new $root.tensorflow.GraphDef();
                if (object.node) {
                    if (!Array.isArray(object.node))
                        throw TypeError(".tensorflow.GraphDef.node: array expected");
                    message.node = [];
                    for (var i = 0; i < object.node.length; ++i) {
                        if (typeof object.node[i] !== "object")
                            throw TypeError(".tensorflow.GraphDef.node: object expected");
                        message.node[i] = $root.tensorflow.NodeDef.fromObject(object.node[i]);
                    }
                }
                if (object.versions != null) {
                    if (typeof object.versions !== "object")
                        throw TypeError(".tensorflow.GraphDef.versions: object expected");
                    message.versions = $root.tensorflow.VersionDef.fromObject(object.versions);
                }
                if (object.version != null)
                    message.version = object.version | 0;
                if (object.library != null) {
                    if (typeof object.library !== "object")
                        throw TypeError(".tensorflow.GraphDef.library: object expected");
                    message.library = $root.tensorflow.FunctionDefLibrary.fromObject(object.library);
                }
                return message;
            };
    
            GraphDef.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.node = [];
                if (options.defaults) {
                    object.library = null;
                    object.version = 0;
                    object.versions = null;
                }
                if (message.node && message.node.length) {
                    object.node = [];
                    for (var j = 0; j < message.node.length; ++j)
                        object.node[j] = $root.tensorflow.NodeDef.toObject(message.node[j], options);
                }
                if (message.library != null && message.hasOwnProperty("library"))
                    object.library = $root.tensorflow.FunctionDefLibrary.toObject(message.library, options);
                if (message.version != null && message.hasOwnProperty("version"))
                    object.version = message.version;
                if (message.versions != null && message.hasOwnProperty("versions"))
                    object.versions = $root.tensorflow.VersionDef.toObject(message.versions, options);
                return object;
            };
    
            GraphDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return GraphDef;
        })();
    
        tensorflow.OpDef = (function() {
    
            function OpDef(properties) {
                this.inputArg = [];
                this.outputArg = [];
                this.attr = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            OpDef.prototype.name = "";
            OpDef.prototype.inputArg = $util.emptyArray;
            OpDef.prototype.outputArg = $util.emptyArray;
            OpDef.prototype.attr = $util.emptyArray;
            OpDef.prototype.deprecation = null;
            OpDef.prototype.summary = "";
            OpDef.prototype.description = "";
            OpDef.prototype.isCommutative = false;
            OpDef.prototype.isAggregate = false;
            OpDef.prototype.isStateful = false;
            OpDef.prototype.allowsUninitializedInput = false;
    
            OpDef.create = function create(properties) {
                return new OpDef(properties);
            };
    
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
                        if (!(message.inputArg && message.inputArg.length))
                            message.inputArg = [];
                        message.inputArg.push($root.tensorflow.OpDef.ArgDef.decode(reader, reader.uint32()));
                        break;
                    case 3:
                        if (!(message.outputArg && message.outputArg.length))
                            message.outputArg = [];
                        message.outputArg.push($root.tensorflow.OpDef.ArgDef.decode(reader, reader.uint32()));
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
                        message.isCommutative = reader.bool();
                        break;
                    case 16:
                        message.isAggregate = reader.bool();
                        break;
                    case 17:
                        message.isStateful = reader.bool();
                        break;
                    case 19:
                        message.allowsUninitializedInput = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            OpDef.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.inputArg != null && message.hasOwnProperty("inputArg")) {
                    if (!Array.isArray(message.inputArg))
                        return "inputArg: array expected";
                    for (var i = 0; i < message.inputArg.length; ++i) {
                        var error = $root.tensorflow.OpDef.ArgDef.verify(message.inputArg[i]);
                        if (error)
                            return "inputArg." + error;
                    }
                }
                if (message.outputArg != null && message.hasOwnProperty("outputArg")) {
                    if (!Array.isArray(message.outputArg))
                        return "outputArg: array expected";
                    for (var i = 0; i < message.outputArg.length; ++i) {
                        var error = $root.tensorflow.OpDef.ArgDef.verify(message.outputArg[i]);
                        if (error)
                            return "outputArg." + error;
                    }
                }
                if (message.attr != null && message.hasOwnProperty("attr")) {
                    if (!Array.isArray(message.attr))
                        return "attr: array expected";
                    for (var i = 0; i < message.attr.length; ++i) {
                        var error = $root.tensorflow.OpDef.AttrDef.verify(message.attr[i]);
                        if (error)
                            return "attr." + error;
                    }
                }
                if (message.deprecation != null && message.hasOwnProperty("deprecation")) {
                    var error = $root.tensorflow.OpDeprecation.verify(message.deprecation);
                    if (error)
                        return "deprecation." + error;
                }
                if (message.summary != null && message.hasOwnProperty("summary"))
                    if (!$util.isString(message.summary))
                        return "summary: string expected";
                if (message.description != null && message.hasOwnProperty("description"))
                    if (!$util.isString(message.description))
                        return "description: string expected";
                if (message.isCommutative != null && message.hasOwnProperty("isCommutative"))
                    if (typeof message.isCommutative !== "boolean")
                        return "isCommutative: boolean expected";
                if (message.isAggregate != null && message.hasOwnProperty("isAggregate"))
                    if (typeof message.isAggregate !== "boolean")
                        return "isAggregate: boolean expected";
                if (message.isStateful != null && message.hasOwnProperty("isStateful"))
                    if (typeof message.isStateful !== "boolean")
                        return "isStateful: boolean expected";
                if (message.allowsUninitializedInput != null && message.hasOwnProperty("allowsUninitializedInput"))
                    if (typeof message.allowsUninitializedInput !== "boolean")
                        return "allowsUninitializedInput: boolean expected";
                return null;
            };
    
            OpDef.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.OpDef)
                    return object;
                var message = new $root.tensorflow.OpDef();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.inputArg) {
                    if (!Array.isArray(object.inputArg))
                        throw TypeError(".tensorflow.OpDef.inputArg: array expected");
                    message.inputArg = [];
                    for (var i = 0; i < object.inputArg.length; ++i) {
                        if (typeof object.inputArg[i] !== "object")
                            throw TypeError(".tensorflow.OpDef.inputArg: object expected");
                        message.inputArg[i] = $root.tensorflow.OpDef.ArgDef.fromObject(object.inputArg[i]);
                    }
                }
                if (object.outputArg) {
                    if (!Array.isArray(object.outputArg))
                        throw TypeError(".tensorflow.OpDef.outputArg: array expected");
                    message.outputArg = [];
                    for (var i = 0; i < object.outputArg.length; ++i) {
                        if (typeof object.outputArg[i] !== "object")
                            throw TypeError(".tensorflow.OpDef.outputArg: object expected");
                        message.outputArg[i] = $root.tensorflow.OpDef.ArgDef.fromObject(object.outputArg[i]);
                    }
                }
                if (object.attr) {
                    if (!Array.isArray(object.attr))
                        throw TypeError(".tensorflow.OpDef.attr: array expected");
                    message.attr = [];
                    for (var i = 0; i < object.attr.length; ++i) {
                        if (typeof object.attr[i] !== "object")
                            throw TypeError(".tensorflow.OpDef.attr: object expected");
                        message.attr[i] = $root.tensorflow.OpDef.AttrDef.fromObject(object.attr[i]);
                    }
                }
                if (object.deprecation != null) {
                    if (typeof object.deprecation !== "object")
                        throw TypeError(".tensorflow.OpDef.deprecation: object expected");
                    message.deprecation = $root.tensorflow.OpDeprecation.fromObject(object.deprecation);
                }
                if (object.summary != null)
                    message.summary = String(object.summary);
                if (object.description != null)
                    message.description = String(object.description);
                if (object.isCommutative != null)
                    message.isCommutative = Boolean(object.isCommutative);
                if (object.isAggregate != null)
                    message.isAggregate = Boolean(object.isAggregate);
                if (object.isStateful != null)
                    message.isStateful = Boolean(object.isStateful);
                if (object.allowsUninitializedInput != null)
                    message.allowsUninitializedInput = Boolean(object.allowsUninitializedInput);
                return message;
            };
    
            OpDef.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.inputArg = [];
                    object.outputArg = [];
                    object.attr = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.summary = "";
                    object.description = "";
                    object.deprecation = null;
                    object.isAggregate = false;
                    object.isStateful = false;
                    object.isCommutative = false;
                    object.allowsUninitializedInput = false;
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.inputArg && message.inputArg.length) {
                    object.inputArg = [];
                    for (var j = 0; j < message.inputArg.length; ++j)
                        object.inputArg[j] = $root.tensorflow.OpDef.ArgDef.toObject(message.inputArg[j], options);
                }
                if (message.outputArg && message.outputArg.length) {
                    object.outputArg = [];
                    for (var j = 0; j < message.outputArg.length; ++j)
                        object.outputArg[j] = $root.tensorflow.OpDef.ArgDef.toObject(message.outputArg[j], options);
                }
                if (message.attr && message.attr.length) {
                    object.attr = [];
                    for (var j = 0; j < message.attr.length; ++j)
                        object.attr[j] = $root.tensorflow.OpDef.AttrDef.toObject(message.attr[j], options);
                }
                if (message.summary != null && message.hasOwnProperty("summary"))
                    object.summary = message.summary;
                if (message.description != null && message.hasOwnProperty("description"))
                    object.description = message.description;
                if (message.deprecation != null && message.hasOwnProperty("deprecation"))
                    object.deprecation = $root.tensorflow.OpDeprecation.toObject(message.deprecation, options);
                if (message.isAggregate != null && message.hasOwnProperty("isAggregate"))
                    object.isAggregate = message.isAggregate;
                if (message.isStateful != null && message.hasOwnProperty("isStateful"))
                    object.isStateful = message.isStateful;
                if (message.isCommutative != null && message.hasOwnProperty("isCommutative"))
                    object.isCommutative = message.isCommutative;
                if (message.allowsUninitializedInput != null && message.hasOwnProperty("allowsUninitializedInput"))
                    object.allowsUninitializedInput = message.allowsUninitializedInput;
                return object;
            };
    
            OpDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
                ArgDef.prototype.typeAttr = "";
                ArgDef.prototype.numberAttr = "";
                ArgDef.prototype.typeListAttr = "";
                ArgDef.prototype.isRef = false;
    
                ArgDef.create = function create(properties) {
                    return new ArgDef(properties);
                };
    
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
                            message.typeAttr = reader.string();
                            break;
                        case 5:
                            message.numberAttr = reader.string();
                            break;
                        case 6:
                            message.typeListAttr = reader.string();
                            break;
                        case 16:
                            message.isRef = reader.bool();
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                ArgDef.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.name != null && message.hasOwnProperty("name"))
                        if (!$util.isString(message.name))
                            return "name: string expected";
                    if (message.description != null && message.hasOwnProperty("description"))
                        if (!$util.isString(message.description))
                            return "description: string expected";
                    if (message.type != null && message.hasOwnProperty("type"))
                        switch (message.type) {
                        default:
                            return "type: enum value expected";
                        case 0:
                        case 1:
                        case 2:
                        case 3:
                        case 4:
                        case 5:
                        case 6:
                        case 7:
                        case 8:
                        case 9:
                        case 10:
                        case 11:
                        case 12:
                        case 13:
                        case 14:
                        case 15:
                        case 16:
                        case 17:
                        case 18:
                        case 19:
                        case 20:
                        case 21:
                        case 22:
                        case 23:
                        case 101:
                        case 102:
                        case 103:
                        case 104:
                        case 105:
                        case 106:
                        case 107:
                        case 108:
                        case 109:
                        case 110:
                        case 111:
                        case 112:
                        case 113:
                        case 114:
                        case 115:
                        case 116:
                        case 117:
                        case 118:
                        case 119:
                        case 120:
                        case 121:
                        case 122:
                        case 123:
                            break;
                        }
                    if (message.typeAttr != null && message.hasOwnProperty("typeAttr"))
                        if (!$util.isString(message.typeAttr))
                            return "typeAttr: string expected";
                    if (message.numberAttr != null && message.hasOwnProperty("numberAttr"))
                        if (!$util.isString(message.numberAttr))
                            return "numberAttr: string expected";
                    if (message.typeListAttr != null && message.hasOwnProperty("typeListAttr"))
                        if (!$util.isString(message.typeListAttr))
                            return "typeListAttr: string expected";
                    if (message.isRef != null && message.hasOwnProperty("isRef"))
                        if (typeof message.isRef !== "boolean")
                            return "isRef: boolean expected";
                    return null;
                };
    
                ArgDef.fromObject = function fromObject(object) {
                    if (object instanceof $root.tensorflow.OpDef.ArgDef)
                        return object;
                    var message = new $root.tensorflow.OpDef.ArgDef();
                    if (object.name != null)
                        message.name = String(object.name);
                    if (object.description != null)
                        message.description = String(object.description);
                    switch (object.type) {
                    case "DT_INVALID":
                    case 0:
                        message.type = 0;
                        break;
                    case "DT_FLOAT":
                    case 1:
                        message.type = 1;
                        break;
                    case "DT_DOUBLE":
                    case 2:
                        message.type = 2;
                        break;
                    case "DT_INT32":
                    case 3:
                        message.type = 3;
                        break;
                    case "DT_UINT8":
                    case 4:
                        message.type = 4;
                        break;
                    case "DT_INT16":
                    case 5:
                        message.type = 5;
                        break;
                    case "DT_INT8":
                    case 6:
                        message.type = 6;
                        break;
                    case "DT_STRING":
                    case 7:
                        message.type = 7;
                        break;
                    case "DT_COMPLEX64":
                    case 8:
                        message.type = 8;
                        break;
                    case "DT_INT64":
                    case 9:
                        message.type = 9;
                        break;
                    case "DT_BOOL":
                    case 10:
                        message.type = 10;
                        break;
                    case "DT_QINT8":
                    case 11:
                        message.type = 11;
                        break;
                    case "DT_QUINT8":
                    case 12:
                        message.type = 12;
                        break;
                    case "DT_QINT32":
                    case 13:
                        message.type = 13;
                        break;
                    case "DT_BFLOAT16":
                    case 14:
                        message.type = 14;
                        break;
                    case "DT_QINT16":
                    case 15:
                        message.type = 15;
                        break;
                    case "DT_QUINT16":
                    case 16:
                        message.type = 16;
                        break;
                    case "DT_UINT16":
                    case 17:
                        message.type = 17;
                        break;
                    case "DT_COMPLEX128":
                    case 18:
                        message.type = 18;
                        break;
                    case "DT_HALF":
                    case 19:
                        message.type = 19;
                        break;
                    case "DT_RESOURCE":
                    case 20:
                        message.type = 20;
                        break;
                    case "DT_VARIANT":
                    case 21:
                        message.type = 21;
                        break;
                    case "DT_UINT32":
                    case 22:
                        message.type = 22;
                        break;
                    case "DT_UINT64":
                    case 23:
                        message.type = 23;
                        break;
                    case "DT_FLOAT_REF":
                    case 101:
                        message.type = 101;
                        break;
                    case "DT_DOUBLE_REF":
                    case 102:
                        message.type = 102;
                        break;
                    case "DT_INT32_REF":
                    case 103:
                        message.type = 103;
                        break;
                    case "DT_UINT8_REF":
                    case 104:
                        message.type = 104;
                        break;
                    case "DT_INT16_REF":
                    case 105:
                        message.type = 105;
                        break;
                    case "DT_INT8_REF":
                    case 106:
                        message.type = 106;
                        break;
                    case "DT_STRING_REF":
                    case 107:
                        message.type = 107;
                        break;
                    case "DT_COMPLEX64_REF":
                    case 108:
                        message.type = 108;
                        break;
                    case "DT_INT64_REF":
                    case 109:
                        message.type = 109;
                        break;
                    case "DT_BOOL_REF":
                    case 110:
                        message.type = 110;
                        break;
                    case "DT_QINT8_REF":
                    case 111:
                        message.type = 111;
                        break;
                    case "DT_QUINT8_REF":
                    case 112:
                        message.type = 112;
                        break;
                    case "DT_QINT32_REF":
                    case 113:
                        message.type = 113;
                        break;
                    case "DT_BFLOAT16_REF":
                    case 114:
                        message.type = 114;
                        break;
                    case "DT_QINT16_REF":
                    case 115:
                        message.type = 115;
                        break;
                    case "DT_QUINT16_REF":
                    case 116:
                        message.type = 116;
                        break;
                    case "DT_UINT16_REF":
                    case 117:
                        message.type = 117;
                        break;
                    case "DT_COMPLEX128_REF":
                    case 118:
                        message.type = 118;
                        break;
                    case "DT_HALF_REF":
                    case 119:
                        message.type = 119;
                        break;
                    case "DT_RESOURCE_REF":
                    case 120:
                        message.type = 120;
                        break;
                    case "DT_VARIANT_REF":
                    case 121:
                        message.type = 121;
                        break;
                    case "DT_UINT32_REF":
                    case 122:
                        message.type = 122;
                        break;
                    case "DT_UINT64_REF":
                    case 123:
                        message.type = 123;
                        break;
                    }
                    if (object.typeAttr != null)
                        message.typeAttr = String(object.typeAttr);
                    if (object.numberAttr != null)
                        message.numberAttr = String(object.numberAttr);
                    if (object.typeListAttr != null)
                        message.typeListAttr = String(object.typeListAttr);
                    if (object.isRef != null)
                        message.isRef = Boolean(object.isRef);
                    return message;
                };
    
                ArgDef.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        object.name = "";
                        object.description = "";
                        object.type = options.enums === String ? "DT_INVALID" : 0;
                        object.typeAttr = "";
                        object.numberAttr = "";
                        object.typeListAttr = "";
                        object.isRef = false;
                    }
                    if (message.name != null && message.hasOwnProperty("name"))
                        object.name = message.name;
                    if (message.description != null && message.hasOwnProperty("description"))
                        object.description = message.description;
                    if (message.type != null && message.hasOwnProperty("type"))
                        object.type = options.enums === String ? $root.tensorflow.DataType[message.type] : message.type;
                    if (message.typeAttr != null && message.hasOwnProperty("typeAttr"))
                        object.typeAttr = message.typeAttr;
                    if (message.numberAttr != null && message.hasOwnProperty("numberAttr"))
                        object.numberAttr = message.numberAttr;
                    if (message.typeListAttr != null && message.hasOwnProperty("typeListAttr"))
                        object.typeListAttr = message.typeListAttr;
                    if (message.isRef != null && message.hasOwnProperty("isRef"))
                        object.isRef = message.isRef;
                    return object;
                };
    
                ArgDef.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
                AttrDef.prototype.defaultValue = null;
                AttrDef.prototype.description = "";
                AttrDef.prototype.hasMinimum = false;
                AttrDef.prototype.minimum = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                AttrDef.prototype.allowedValues = null;
    
                AttrDef.create = function create(properties) {
                    return new AttrDef(properties);
                };
    
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
                            message.defaultValue = $root.tensorflow.AttrValue.decode(reader, reader.uint32());
                            break;
                        case 4:
                            message.description = reader.string();
                            break;
                        case 5:
                            message.hasMinimum = reader.bool();
                            break;
                        case 6:
                            message.minimum = reader.int64();
                            break;
                        case 7:
                            message.allowedValues = $root.tensorflow.AttrValue.decode(reader, reader.uint32());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                AttrDef.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.name != null && message.hasOwnProperty("name"))
                        if (!$util.isString(message.name))
                            return "name: string expected";
                    if (message.type != null && message.hasOwnProperty("type"))
                        if (!$util.isString(message.type))
                            return "type: string expected";
                    if (message.defaultValue != null && message.hasOwnProperty("defaultValue")) {
                        var error = $root.tensorflow.AttrValue.verify(message.defaultValue);
                        if (error)
                            return "defaultValue." + error;
                    }
                    if (message.description != null && message.hasOwnProperty("description"))
                        if (!$util.isString(message.description))
                            return "description: string expected";
                    if (message.hasMinimum != null && message.hasOwnProperty("hasMinimum"))
                        if (typeof message.hasMinimum !== "boolean")
                            return "hasMinimum: boolean expected";
                    if (message.minimum != null && message.hasOwnProperty("minimum"))
                        if (!$util.isInteger(message.minimum) && !(message.minimum && $util.isInteger(message.minimum.low) && $util.isInteger(message.minimum.high)))
                            return "minimum: integer|Long expected";
                    if (message.allowedValues != null && message.hasOwnProperty("allowedValues")) {
                        var error = $root.tensorflow.AttrValue.verify(message.allowedValues);
                        if (error)
                            return "allowedValues." + error;
                    }
                    return null;
                };
    
                AttrDef.fromObject = function fromObject(object) {
                    if (object instanceof $root.tensorflow.OpDef.AttrDef)
                        return object;
                    var message = new $root.tensorflow.OpDef.AttrDef();
                    if (object.name != null)
                        message.name = String(object.name);
                    if (object.type != null)
                        message.type = String(object.type);
                    if (object.defaultValue != null) {
                        if (typeof object.defaultValue !== "object")
                            throw TypeError(".tensorflow.OpDef.AttrDef.defaultValue: object expected");
                        message.defaultValue = $root.tensorflow.AttrValue.fromObject(object.defaultValue);
                    }
                    if (object.description != null)
                        message.description = String(object.description);
                    if (object.hasMinimum != null)
                        message.hasMinimum = Boolean(object.hasMinimum);
                    if (object.minimum != null)
                        if ($util.Long)
                            (message.minimum = $util.Long.fromValue(object.minimum)).unsigned = false;
                        else if (typeof object.minimum === "string")
                            message.minimum = parseInt(object.minimum, 10);
                        else if (typeof object.minimum === "number")
                            message.minimum = object.minimum;
                        else if (typeof object.minimum === "object")
                            message.minimum = new $util.LongBits(object.minimum.low >>> 0, object.minimum.high >>> 0).toNumber();
                    if (object.allowedValues != null) {
                        if (typeof object.allowedValues !== "object")
                            throw TypeError(".tensorflow.OpDef.AttrDef.allowedValues: object expected");
                        message.allowedValues = $root.tensorflow.AttrValue.fromObject(object.allowedValues);
                    }
                    return message;
                };
    
                AttrDef.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        object.name = "";
                        object.type = "";
                        object.defaultValue = null;
                        object.description = "";
                        object.hasMinimum = false;
                        if ($util.Long) {
                            var long = new $util.Long(0, 0, false);
                            object.minimum = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                        } else
                            object.minimum = options.longs === String ? "0" : 0;
                        object.allowedValues = null;
                    }
                    if (message.name != null && message.hasOwnProperty("name"))
                        object.name = message.name;
                    if (message.type != null && message.hasOwnProperty("type"))
                        object.type = message.type;
                    if (message.defaultValue != null && message.hasOwnProperty("defaultValue"))
                        object.defaultValue = $root.tensorflow.AttrValue.toObject(message.defaultValue, options);
                    if (message.description != null && message.hasOwnProperty("description"))
                        object.description = message.description;
                    if (message.hasMinimum != null && message.hasOwnProperty("hasMinimum"))
                        object.hasMinimum = message.hasMinimum;
                    if (message.minimum != null && message.hasOwnProperty("minimum"))
                        if (typeof message.minimum === "number")
                            object.minimum = options.longs === String ? String(message.minimum) : message.minimum;
                        else
                            object.minimum = options.longs === String ? $util.Long.prototype.toString.call(message.minimum) : options.longs === Number ? new $util.LongBits(message.minimum.low >>> 0, message.minimum.high >>> 0).toNumber() : message.minimum;
                    if (message.allowedValues != null && message.hasOwnProperty("allowedValues"))
                        object.allowedValues = $root.tensorflow.AttrValue.toObject(message.allowedValues, options);
                    return object;
                };
    
                AttrDef.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
            OpDeprecation.create = function create(properties) {
                return new OpDeprecation(properties);
            };
    
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
    
            OpDeprecation.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.version != null && message.hasOwnProperty("version"))
                    if (!$util.isInteger(message.version))
                        return "version: integer expected";
                if (message.explanation != null && message.hasOwnProperty("explanation"))
                    if (!$util.isString(message.explanation))
                        return "explanation: string expected";
                return null;
            };
    
            OpDeprecation.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.OpDeprecation)
                    return object;
                var message = new $root.tensorflow.OpDeprecation();
                if (object.version != null)
                    message.version = object.version | 0;
                if (object.explanation != null)
                    message.explanation = String(object.explanation);
                return message;
            };
    
            OpDeprecation.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.version = 0;
                    object.explanation = "";
                }
                if (message.version != null && message.hasOwnProperty("version"))
                    object.version = message.version;
                if (message.explanation != null && message.hasOwnProperty("explanation"))
                    object.explanation = message.explanation;
                return object;
            };
    
            OpDeprecation.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
            OpList.create = function create(properties) {
                return new OpList(properties);
            };
    
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
    
            OpList.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.op != null && message.hasOwnProperty("op")) {
                    if (!Array.isArray(message.op))
                        return "op: array expected";
                    for (var i = 0; i < message.op.length; ++i) {
                        var error = $root.tensorflow.OpDef.verify(message.op[i]);
                        if (error)
                            return "op." + error;
                    }
                }
                return null;
            };
    
            OpList.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.OpList)
                    return object;
                var message = new $root.tensorflow.OpList();
                if (object.op) {
                    if (!Array.isArray(object.op))
                        throw TypeError(".tensorflow.OpList.op: array expected");
                    message.op = [];
                    for (var i = 0; i < object.op.length; ++i) {
                        if (typeof object.op[i] !== "object")
                            throw TypeError(".tensorflow.OpList.op: object expected");
                        message.op[i] = $root.tensorflow.OpDef.fromObject(object.op[i]);
                    }
                }
                return message;
            };
    
            OpList.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.op = [];
                if (message.op && message.op.length) {
                    object.op = [];
                    for (var j = 0; j < message.op.length; ++j)
                        object.op[j] = $root.tensorflow.OpDef.toObject(message.op[j], options);
                }
                return object;
            };
    
            OpList.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
            TensorShapeProto.prototype.unknownRank = false;
    
            TensorShapeProto.create = function create(properties) {
                return new TensorShapeProto(properties);
            };
    
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
                        message.unknownRank = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TensorShapeProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.dim != null && message.hasOwnProperty("dim")) {
                    if (!Array.isArray(message.dim))
                        return "dim: array expected";
                    for (var i = 0; i < message.dim.length; ++i) {
                        var error = $root.tensorflow.TensorShapeProto.Dim.verify(message.dim[i]);
                        if (error)
                            return "dim." + error;
                    }
                }
                if (message.unknownRank != null && message.hasOwnProperty("unknownRank"))
                    if (typeof message.unknownRank !== "boolean")
                        return "unknownRank: boolean expected";
                return null;
            };
    
            TensorShapeProto.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.TensorShapeProto)
                    return object;
                var message = new $root.tensorflow.TensorShapeProto();
                if (object.dim) {
                    if (!Array.isArray(object.dim))
                        throw TypeError(".tensorflow.TensorShapeProto.dim: array expected");
                    message.dim = [];
                    for (var i = 0; i < object.dim.length; ++i) {
                        if (typeof object.dim[i] !== "object")
                            throw TypeError(".tensorflow.TensorShapeProto.dim: object expected");
                        message.dim[i] = $root.tensorflow.TensorShapeProto.Dim.fromObject(object.dim[i]);
                    }
                }
                if (object.unknownRank != null)
                    message.unknownRank = Boolean(object.unknownRank);
                return message;
            };
    
            TensorShapeProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.dim = [];
                if (options.defaults)
                    object.unknownRank = false;
                if (message.dim && message.dim.length) {
                    object.dim = [];
                    for (var j = 0; j < message.dim.length; ++j)
                        object.dim[j] = $root.tensorflow.TensorShapeProto.Dim.toObject(message.dim[j], options);
                }
                if (message.unknownRank != null && message.hasOwnProperty("unknownRank"))
                    object.unknownRank = message.unknownRank;
                return object;
            };
    
            TensorShapeProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                Dim.create = function create(properties) {
                    return new Dim(properties);
                };
    
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
    
                Dim.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.size != null && message.hasOwnProperty("size"))
                        if (!$util.isInteger(message.size) && !(message.size && $util.isInteger(message.size.low) && $util.isInteger(message.size.high)))
                            return "size: integer|Long expected";
                    if (message.name != null && message.hasOwnProperty("name"))
                        if (!$util.isString(message.name))
                            return "name: string expected";
                    return null;
                };
    
                Dim.fromObject = function fromObject(object) {
                    if (object instanceof $root.tensorflow.TensorShapeProto.Dim)
                        return object;
                    var message = new $root.tensorflow.TensorShapeProto.Dim();
                    if (object.size != null)
                        if ($util.Long)
                            (message.size = $util.Long.fromValue(object.size)).unsigned = false;
                        else if (typeof object.size === "string")
                            message.size = parseInt(object.size, 10);
                        else if (typeof object.size === "number")
                            message.size = object.size;
                        else if (typeof object.size === "object")
                            message.size = new $util.LongBits(object.size.low >>> 0, object.size.high >>> 0).toNumber();
                    if (object.name != null)
                        message.name = String(object.name);
                    return message;
                };
    
                Dim.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        if ($util.Long) {
                            var long = new $util.Long(0, 0, false);
                            object.size = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                        } else
                            object.size = options.longs === String ? "0" : 0;
                        object.name = "";
                    }
                    if (message.size != null && message.hasOwnProperty("size"))
                        if (typeof message.size === "number")
                            object.size = options.longs === String ? String(message.size) : message.size;
                        else
                            object.size = options.longs === String ? $util.Long.prototype.toString.call(message.size) : options.longs === Number ? new $util.LongBits(message.size.low >>> 0, message.size.high >>> 0).toNumber() : message.size;
                    if (message.name != null && message.hasOwnProperty("name"))
                        object.name = message.name;
                    return object;
                };
    
                Dim.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
            NodeDef.create = function create(properties) {
                return new NodeDef(properties);
            };
    
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
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            NodeDef.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.op != null && message.hasOwnProperty("op"))
                    if (!$util.isString(message.op))
                        return "op: string expected";
                if (message.input != null && message.hasOwnProperty("input")) {
                    if (!Array.isArray(message.input))
                        return "input: array expected";
                    for (var i = 0; i < message.input.length; ++i)
                        if (!$util.isString(message.input[i]))
                            return "input: string[] expected";
                }
                if (message.device != null && message.hasOwnProperty("device"))
                    if (!$util.isString(message.device))
                        return "device: string expected";
                if (message.attr != null && message.hasOwnProperty("attr")) {
                    if (!$util.isObject(message.attr))
                        return "attr: object expected";
                    var key = Object.keys(message.attr);
                    for (var i = 0; i < key.length; ++i) {
                        var error = $root.tensorflow.AttrValue.verify(message.attr[key[i]]);
                        if (error)
                            return "attr." + error;
                    }
                }
                return null;
            };
    
            NodeDef.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.NodeDef)
                    return object;
                var message = new $root.tensorflow.NodeDef();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.op != null)
                    message.op = String(object.op);
                if (object.input) {
                    if (!Array.isArray(object.input))
                        throw TypeError(".tensorflow.NodeDef.input: array expected");
                    message.input = [];
                    for (var i = 0; i < object.input.length; ++i)
                        message.input[i] = String(object.input[i]);
                }
                if (object.device != null)
                    message.device = String(object.device);
                if (object.attr) {
                    if (typeof object.attr !== "object")
                        throw TypeError(".tensorflow.NodeDef.attr: object expected");
                    message.attr = {};
                    for (var keys = Object.keys(object.attr), i = 0; i < keys.length; ++i) {
                        if (typeof object.attr[keys[i]] !== "object")
                            throw TypeError(".tensorflow.NodeDef.attr: object expected");
                        message.attr[keys[i]] = $root.tensorflow.AttrValue.fromObject(object.attr[keys[i]]);
                    }
                }
                return message;
            };
    
            NodeDef.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.input = [];
                if (options.objects || options.defaults)
                    object.attr = {};
                if (options.defaults) {
                    object.name = "";
                    object.op = "";
                    object.device = "";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.op != null && message.hasOwnProperty("op"))
                    object.op = message.op;
                if (message.input && message.input.length) {
                    object.input = [];
                    for (var j = 0; j < message.input.length; ++j)
                        object.input[j] = message.input[j];
                }
                if (message.device != null && message.hasOwnProperty("device"))
                    object.device = message.device;
                var keys2;
                if (message.attr && (keys2 = Object.keys(message.attr)).length) {
                    object.attr = {};
                    for (var j = 0; j < keys2.length; ++j)
                        object.attr[keys2[j]] = $root.tensorflow.AttrValue.toObject(message.attr[keys2[j]], options);
                }
                return object;
            };
    
            NodeDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return NodeDef;
        })();
    
        tensorflow.VersionDef = (function() {
    
            function VersionDef(properties) {
                this.badConsumers = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            VersionDef.prototype.producer = 0;
            VersionDef.prototype.minConsumer = 0;
            VersionDef.prototype.badConsumers = $util.emptyArray;
    
            VersionDef.create = function create(properties) {
                return new VersionDef(properties);
            };
    
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
                        message.minConsumer = reader.int32();
                        break;
                    case 3:
                        if (!(message.badConsumers && message.badConsumers.length))
                            message.badConsumers = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.badConsumers.push(reader.int32());
                        } else
                            message.badConsumers.push(reader.int32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            VersionDef.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.producer != null && message.hasOwnProperty("producer"))
                    if (!$util.isInteger(message.producer))
                        return "producer: integer expected";
                if (message.minConsumer != null && message.hasOwnProperty("minConsumer"))
                    if (!$util.isInteger(message.minConsumer))
                        return "minConsumer: integer expected";
                if (message.badConsumers != null && message.hasOwnProperty("badConsumers")) {
                    if (!Array.isArray(message.badConsumers))
                        return "badConsumers: array expected";
                    for (var i = 0; i < message.badConsumers.length; ++i)
                        if (!$util.isInteger(message.badConsumers[i]))
                            return "badConsumers: integer[] expected";
                }
                return null;
            };
    
            VersionDef.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.VersionDef)
                    return object;
                var message = new $root.tensorflow.VersionDef();
                if (object.producer != null)
                    message.producer = object.producer | 0;
                if (object.minConsumer != null)
                    message.minConsumer = object.minConsumer | 0;
                if (object.badConsumers) {
                    if (!Array.isArray(object.badConsumers))
                        throw TypeError(".tensorflow.VersionDef.badConsumers: array expected");
                    message.badConsumers = [];
                    for (var i = 0; i < object.badConsumers.length; ++i)
                        message.badConsumers[i] = object.badConsumers[i] | 0;
                }
                return message;
            };
    
            VersionDef.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.badConsumers = [];
                if (options.defaults) {
                    object.producer = 0;
                    object.minConsumer = 0;
                }
                if (message.producer != null && message.hasOwnProperty("producer"))
                    object.producer = message.producer;
                if (message.minConsumer != null && message.hasOwnProperty("minConsumer"))
                    object.minConsumer = message.minConsumer;
                if (message.badConsumers && message.badConsumers.length) {
                    object.badConsumers = [];
                    for (var j = 0; j < message.badConsumers.length; ++j)
                        object.badConsumers[j] = message.badConsumers[j];
                }
                return object;
            };
    
            VersionDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
            FunctionDefLibrary.create = function create(properties) {
                return new FunctionDefLibrary(properties);
            };
    
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
    
            FunctionDefLibrary.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message["function"] != null && message.hasOwnProperty("function")) {
                    if (!Array.isArray(message["function"]))
                        return "function: array expected";
                    for (var i = 0; i < message["function"].length; ++i) {
                        var error = $root.tensorflow.FunctionDef.verify(message["function"][i]);
                        if (error)
                            return "function." + error;
                    }
                }
                if (message.gradient != null && message.hasOwnProperty("gradient")) {
                    if (!Array.isArray(message.gradient))
                        return "gradient: array expected";
                    for (var i = 0; i < message.gradient.length; ++i) {
                        var error = $root.tensorflow.GradientDef.verify(message.gradient[i]);
                        if (error)
                            return "gradient." + error;
                    }
                }
                return null;
            };
    
            FunctionDefLibrary.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.FunctionDefLibrary)
                    return object;
                var message = new $root.tensorflow.FunctionDefLibrary();
                if (object["function"]) {
                    if (!Array.isArray(object["function"]))
                        throw TypeError(".tensorflow.FunctionDefLibrary.function: array expected");
                    message["function"] = [];
                    for (var i = 0; i < object["function"].length; ++i) {
                        if (typeof object["function"][i] !== "object")
                            throw TypeError(".tensorflow.FunctionDefLibrary.function: object expected");
                        message["function"][i] = $root.tensorflow.FunctionDef.fromObject(object["function"][i]);
                    }
                }
                if (object.gradient) {
                    if (!Array.isArray(object.gradient))
                        throw TypeError(".tensorflow.FunctionDefLibrary.gradient: array expected");
                    message.gradient = [];
                    for (var i = 0; i < object.gradient.length; ++i) {
                        if (typeof object.gradient[i] !== "object")
                            throw TypeError(".tensorflow.FunctionDefLibrary.gradient: object expected");
                        message.gradient[i] = $root.tensorflow.GradientDef.fromObject(object.gradient[i]);
                    }
                }
                return message;
            };
    
            FunctionDefLibrary.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object["function"] = [];
                    object.gradient = [];
                }
                if (message["function"] && message["function"].length) {
                    object["function"] = [];
                    for (var j = 0; j < message["function"].length; ++j)
                        object["function"][j] = $root.tensorflow.FunctionDef.toObject(message["function"][j], options);
                }
                if (message.gradient && message.gradient.length) {
                    object.gradient = [];
                    for (var j = 0; j < message.gradient.length; ++j)
                        object.gradient[j] = $root.tensorflow.GradientDef.toObject(message.gradient[j], options);
                }
                return object;
            };
    
            FunctionDefLibrary.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return FunctionDefLibrary;
        })();
    
        tensorflow.FunctionDef = (function() {
    
            function FunctionDef(properties) {
                this.attr = {};
                this.nodeDef = [];
                this.ret = {};
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            FunctionDef.prototype.signature = null;
            FunctionDef.prototype.attr = $util.emptyObject;
            FunctionDef.prototype.nodeDef = $util.emptyArray;
            FunctionDef.prototype.ret = $util.emptyObject;
    
            FunctionDef.create = function create(properties) {
                return new FunctionDef(properties);
            };
    
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
                    case 3:
                        if (!(message.nodeDef && message.nodeDef.length))
                            message.nodeDef = [];
                        message.nodeDef.push($root.tensorflow.NodeDef.decode(reader, reader.uint32()));
                        break;
                    case 4:
                        reader.skip().pos++;
                        if (message.ret === $util.emptyObject)
                            message.ret = {};
                        key = reader.string();
                        reader.pos++;
                        message.ret[key] = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            FunctionDef.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.signature != null && message.hasOwnProperty("signature")) {
                    var error = $root.tensorflow.OpDef.verify(message.signature);
                    if (error)
                        return "signature." + error;
                }
                if (message.attr != null && message.hasOwnProperty("attr")) {
                    if (!$util.isObject(message.attr))
                        return "attr: object expected";
                    var key = Object.keys(message.attr);
                    for (var i = 0; i < key.length; ++i) {
                        var error = $root.tensorflow.AttrValue.verify(message.attr[key[i]]);
                        if (error)
                            return "attr." + error;
                    }
                }
                if (message.nodeDef != null && message.hasOwnProperty("nodeDef")) {
                    if (!Array.isArray(message.nodeDef))
                        return "nodeDef: array expected";
                    for (var i = 0; i < message.nodeDef.length; ++i) {
                        var error = $root.tensorflow.NodeDef.verify(message.nodeDef[i]);
                        if (error)
                            return "nodeDef." + error;
                    }
                }
                if (message.ret != null && message.hasOwnProperty("ret")) {
                    if (!$util.isObject(message.ret))
                        return "ret: object expected";
                    var key = Object.keys(message.ret);
                    for (var i = 0; i < key.length; ++i)
                        if (!$util.isString(message.ret[key[i]]))
                            return "ret: string{k:string} expected";
                }
                return null;
            };
    
            FunctionDef.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.FunctionDef)
                    return object;
                var message = new $root.tensorflow.FunctionDef();
                if (object.signature != null) {
                    if (typeof object.signature !== "object")
                        throw TypeError(".tensorflow.FunctionDef.signature: object expected");
                    message.signature = $root.tensorflow.OpDef.fromObject(object.signature);
                }
                if (object.attr) {
                    if (typeof object.attr !== "object")
                        throw TypeError(".tensorflow.FunctionDef.attr: object expected");
                    message.attr = {};
                    for (var keys = Object.keys(object.attr), i = 0; i < keys.length; ++i) {
                        if (typeof object.attr[keys[i]] !== "object")
                            throw TypeError(".tensorflow.FunctionDef.attr: object expected");
                        message.attr[keys[i]] = $root.tensorflow.AttrValue.fromObject(object.attr[keys[i]]);
                    }
                }
                if (object.nodeDef) {
                    if (!Array.isArray(object.nodeDef))
                        throw TypeError(".tensorflow.FunctionDef.nodeDef: array expected");
                    message.nodeDef = [];
                    for (var i = 0; i < object.nodeDef.length; ++i) {
                        if (typeof object.nodeDef[i] !== "object")
                            throw TypeError(".tensorflow.FunctionDef.nodeDef: object expected");
                        message.nodeDef[i] = $root.tensorflow.NodeDef.fromObject(object.nodeDef[i]);
                    }
                }
                if (object.ret) {
                    if (typeof object.ret !== "object")
                        throw TypeError(".tensorflow.FunctionDef.ret: object expected");
                    message.ret = {};
                    for (var keys = Object.keys(object.ret), i = 0; i < keys.length; ++i)
                        message.ret[keys[i]] = String(object.ret[keys[i]]);
                }
                return message;
            };
    
            FunctionDef.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.nodeDef = [];
                if (options.objects || options.defaults) {
                    object.ret = {};
                    object.attr = {};
                }
                if (options.defaults)
                    object.signature = null;
                if (message.signature != null && message.hasOwnProperty("signature"))
                    object.signature = $root.tensorflow.OpDef.toObject(message.signature, options);
                if (message.nodeDef && message.nodeDef.length) {
                    object.nodeDef = [];
                    for (var j = 0; j < message.nodeDef.length; ++j)
                        object.nodeDef[j] = $root.tensorflow.NodeDef.toObject(message.nodeDef[j], options);
                }
                var keys2;
                if (message.ret && (keys2 = Object.keys(message.ret)).length) {
                    object.ret = {};
                    for (var j = 0; j < keys2.length; ++j)
                        object.ret[keys2[j]] = message.ret[keys2[j]];
                }
                if (message.attr && (keys2 = Object.keys(message.attr)).length) {
                    object.attr = {};
                    for (var j = 0; j < keys2.length; ++j)
                        object.attr[keys2[j]] = $root.tensorflow.AttrValue.toObject(message.attr[keys2[j]], options);
                }
                return object;
            };
    
            FunctionDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return FunctionDef;
        })();
    
        tensorflow.GradientDef = (function() {
    
            function GradientDef(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            GradientDef.prototype.functionName = "";
            GradientDef.prototype.gradientFunc = "";
    
            GradientDef.create = function create(properties) {
                return new GradientDef(properties);
            };
    
            GradientDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.GradientDef();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.functionName = reader.string();
                        break;
                    case 2:
                        message.gradientFunc = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            GradientDef.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.functionName != null && message.hasOwnProperty("functionName"))
                    if (!$util.isString(message.functionName))
                        return "functionName: string expected";
                if (message.gradientFunc != null && message.hasOwnProperty("gradientFunc"))
                    if (!$util.isString(message.gradientFunc))
                        return "gradientFunc: string expected";
                return null;
            };
    
            GradientDef.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.GradientDef)
                    return object;
                var message = new $root.tensorflow.GradientDef();
                if (object.functionName != null)
                    message.functionName = String(object.functionName);
                if (object.gradientFunc != null)
                    message.gradientFunc = String(object.gradientFunc);
                return message;
            };
    
            GradientDef.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.functionName = "";
                    object.gradientFunc = "";
                }
                if (message.functionName != null && message.hasOwnProperty("functionName"))
                    object.functionName = message.functionName;
                if (message.gradientFunc != null && message.hasOwnProperty("gradientFunc"))
                    object.gradientFunc = message.gradientFunc;
                return object;
            };
    
            GradientDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
            AttrValue.create = function create(properties) {
                return new AttrValue(properties);
            };
    
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
    
            AttrValue.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                var properties = {};
                if (message.s != null && message.hasOwnProperty("s")) {
                    properties.value = 1;
                    if (!(message.s && typeof message.s.length === "number" || $util.isString(message.s)))
                        return "s: buffer expected";
                }
                if (message.i != null && message.hasOwnProperty("i")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    if (!$util.isInteger(message.i) && !(message.i && $util.isInteger(message.i.low) && $util.isInteger(message.i.high)))
                        return "i: integer|Long expected";
                }
                if (message.f != null && message.hasOwnProperty("f")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    if (typeof message.f !== "number")
                        return "f: number expected";
                }
                if (message.b != null && message.hasOwnProperty("b")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    if (typeof message.b !== "boolean")
                        return "b: boolean expected";
                }
                if (message.type != null && message.hasOwnProperty("type")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    switch (message.type) {
                    default:
                        return "type: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                    case 3:
                    case 4:
                    case 5:
                    case 6:
                    case 7:
                    case 8:
                    case 9:
                    case 10:
                    case 11:
                    case 12:
                    case 13:
                    case 14:
                    case 15:
                    case 16:
                    case 17:
                    case 18:
                    case 19:
                    case 20:
                    case 21:
                    case 22:
                    case 23:
                    case 101:
                    case 102:
                    case 103:
                    case 104:
                    case 105:
                    case 106:
                    case 107:
                    case 108:
                    case 109:
                    case 110:
                    case 111:
                    case 112:
                    case 113:
                    case 114:
                    case 115:
                    case 116:
                    case 117:
                    case 118:
                    case 119:
                    case 120:
                    case 121:
                    case 122:
                    case 123:
                        break;
                    }
                }
                if (message.shape != null && message.hasOwnProperty("shape")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    {
                        var error = $root.tensorflow.TensorShapeProto.verify(message.shape);
                        if (error)
                            return "shape." + error;
                    }
                }
                if (message.tensor != null && message.hasOwnProperty("tensor")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    {
                        var error = $root.tensorflow.TensorProto.verify(message.tensor);
                        if (error)
                            return "tensor." + error;
                    }
                }
                if (message.list != null && message.hasOwnProperty("list")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    {
                        var error = $root.tensorflow.AttrValue.ListValue.verify(message.list);
                        if (error)
                            return "list." + error;
                    }
                }
                if (message.func != null && message.hasOwnProperty("func")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    {
                        var error = $root.tensorflow.NameAttrList.verify(message.func);
                        if (error)
                            return "func." + error;
                    }
                }
                if (message.placeholder != null && message.hasOwnProperty("placeholder")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    if (!$util.isString(message.placeholder))
                        return "placeholder: string expected";
                }
                return null;
            };
    
            AttrValue.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.AttrValue)
                    return object;
                var message = new $root.tensorflow.AttrValue();
                if (object.s != null)
                    if (typeof object.s === "string")
                        $util.base64.decode(object.s, message.s = $util.newBuffer($util.base64.length(object.s)), 0);
                    else if (object.s.length)
                        message.s = object.s;
                if (object.i != null)
                    if ($util.Long)
                        (message.i = $util.Long.fromValue(object.i)).unsigned = false;
                    else if (typeof object.i === "string")
                        message.i = parseInt(object.i, 10);
                    else if (typeof object.i === "number")
                        message.i = object.i;
                    else if (typeof object.i === "object")
                        message.i = new $util.LongBits(object.i.low >>> 0, object.i.high >>> 0).toNumber();
                if (object.f != null)
                    message.f = Number(object.f);
                if (object.b != null)
                    message.b = Boolean(object.b);
                switch (object.type) {
                case "DT_INVALID":
                case 0:
                    message.type = 0;
                    break;
                case "DT_FLOAT":
                case 1:
                    message.type = 1;
                    break;
                case "DT_DOUBLE":
                case 2:
                    message.type = 2;
                    break;
                case "DT_INT32":
                case 3:
                    message.type = 3;
                    break;
                case "DT_UINT8":
                case 4:
                    message.type = 4;
                    break;
                case "DT_INT16":
                case 5:
                    message.type = 5;
                    break;
                case "DT_INT8":
                case 6:
                    message.type = 6;
                    break;
                case "DT_STRING":
                case 7:
                    message.type = 7;
                    break;
                case "DT_COMPLEX64":
                case 8:
                    message.type = 8;
                    break;
                case "DT_INT64":
                case 9:
                    message.type = 9;
                    break;
                case "DT_BOOL":
                case 10:
                    message.type = 10;
                    break;
                case "DT_QINT8":
                case 11:
                    message.type = 11;
                    break;
                case "DT_QUINT8":
                case 12:
                    message.type = 12;
                    break;
                case "DT_QINT32":
                case 13:
                    message.type = 13;
                    break;
                case "DT_BFLOAT16":
                case 14:
                    message.type = 14;
                    break;
                case "DT_QINT16":
                case 15:
                    message.type = 15;
                    break;
                case "DT_QUINT16":
                case 16:
                    message.type = 16;
                    break;
                case "DT_UINT16":
                case 17:
                    message.type = 17;
                    break;
                case "DT_COMPLEX128":
                case 18:
                    message.type = 18;
                    break;
                case "DT_HALF":
                case 19:
                    message.type = 19;
                    break;
                case "DT_RESOURCE":
                case 20:
                    message.type = 20;
                    break;
                case "DT_VARIANT":
                case 21:
                    message.type = 21;
                    break;
                case "DT_UINT32":
                case 22:
                    message.type = 22;
                    break;
                case "DT_UINT64":
                case 23:
                    message.type = 23;
                    break;
                case "DT_FLOAT_REF":
                case 101:
                    message.type = 101;
                    break;
                case "DT_DOUBLE_REF":
                case 102:
                    message.type = 102;
                    break;
                case "DT_INT32_REF":
                case 103:
                    message.type = 103;
                    break;
                case "DT_UINT8_REF":
                case 104:
                    message.type = 104;
                    break;
                case "DT_INT16_REF":
                case 105:
                    message.type = 105;
                    break;
                case "DT_INT8_REF":
                case 106:
                    message.type = 106;
                    break;
                case "DT_STRING_REF":
                case 107:
                    message.type = 107;
                    break;
                case "DT_COMPLEX64_REF":
                case 108:
                    message.type = 108;
                    break;
                case "DT_INT64_REF":
                case 109:
                    message.type = 109;
                    break;
                case "DT_BOOL_REF":
                case 110:
                    message.type = 110;
                    break;
                case "DT_QINT8_REF":
                case 111:
                    message.type = 111;
                    break;
                case "DT_QUINT8_REF":
                case 112:
                    message.type = 112;
                    break;
                case "DT_QINT32_REF":
                case 113:
                    message.type = 113;
                    break;
                case "DT_BFLOAT16_REF":
                case 114:
                    message.type = 114;
                    break;
                case "DT_QINT16_REF":
                case 115:
                    message.type = 115;
                    break;
                case "DT_QUINT16_REF":
                case 116:
                    message.type = 116;
                    break;
                case "DT_UINT16_REF":
                case 117:
                    message.type = 117;
                    break;
                case "DT_COMPLEX128_REF":
                case 118:
                    message.type = 118;
                    break;
                case "DT_HALF_REF":
                case 119:
                    message.type = 119;
                    break;
                case "DT_RESOURCE_REF":
                case 120:
                    message.type = 120;
                    break;
                case "DT_VARIANT_REF":
                case 121:
                    message.type = 121;
                    break;
                case "DT_UINT32_REF":
                case 122:
                    message.type = 122;
                    break;
                case "DT_UINT64_REF":
                case 123:
                    message.type = 123;
                    break;
                }
                if (object.shape != null) {
                    if (typeof object.shape !== "object")
                        throw TypeError(".tensorflow.AttrValue.shape: object expected");
                    message.shape = $root.tensorflow.TensorShapeProto.fromObject(object.shape);
                }
                if (object.tensor != null) {
                    if (typeof object.tensor !== "object")
                        throw TypeError(".tensorflow.AttrValue.tensor: object expected");
                    message.tensor = $root.tensorflow.TensorProto.fromObject(object.tensor);
                }
                if (object.list != null) {
                    if (typeof object.list !== "object")
                        throw TypeError(".tensorflow.AttrValue.list: object expected");
                    message.list = $root.tensorflow.AttrValue.ListValue.fromObject(object.list);
                }
                if (object.func != null) {
                    if (typeof object.func !== "object")
                        throw TypeError(".tensorflow.AttrValue.func: object expected");
                    message.func = $root.tensorflow.NameAttrList.fromObject(object.func);
                }
                if (object.placeholder != null)
                    message.placeholder = String(object.placeholder);
                return message;
            };
    
            AttrValue.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (message.list != null && message.hasOwnProperty("list")) {
                    object.list = $root.tensorflow.AttrValue.ListValue.toObject(message.list, options);
                    if (options.oneofs)
                        object.value = "list";
                }
                if (message.s != null && message.hasOwnProperty("s")) {
                    object.s = options.bytes === String ? $util.base64.encode(message.s, 0, message.s.length) : options.bytes === Array ? Array.prototype.slice.call(message.s) : message.s;
                    if (options.oneofs)
                        object.value = "s";
                }
                if (message.i != null && message.hasOwnProperty("i")) {
                    if (typeof message.i === "number")
                        object.i = options.longs === String ? String(message.i) : message.i;
                    else
                        object.i = options.longs === String ? $util.Long.prototype.toString.call(message.i) : options.longs === Number ? new $util.LongBits(message.i.low >>> 0, message.i.high >>> 0).toNumber() : message.i;
                    if (options.oneofs)
                        object.value = "i";
                }
                if (message.f != null && message.hasOwnProperty("f")) {
                    object.f = options.json && !isFinite(message.f) ? String(message.f) : message.f;
                    if (options.oneofs)
                        object.value = "f";
                }
                if (message.b != null && message.hasOwnProperty("b")) {
                    object.b = message.b;
                    if (options.oneofs)
                        object.value = "b";
                }
                if (message.type != null && message.hasOwnProperty("type")) {
                    object.type = options.enums === String ? $root.tensorflow.DataType[message.type] : message.type;
                    if (options.oneofs)
                        object.value = "type";
                }
                if (message.shape != null && message.hasOwnProperty("shape")) {
                    object.shape = $root.tensorflow.TensorShapeProto.toObject(message.shape, options);
                    if (options.oneofs)
                        object.value = "shape";
                }
                if (message.tensor != null && message.hasOwnProperty("tensor")) {
                    object.tensor = $root.tensorflow.TensorProto.toObject(message.tensor, options);
                    if (options.oneofs)
                        object.value = "tensor";
                }
                if (message.placeholder != null && message.hasOwnProperty("placeholder")) {
                    object.placeholder = message.placeholder;
                    if (options.oneofs)
                        object.value = "placeholder";
                }
                if (message.func != null && message.hasOwnProperty("func")) {
                    object.func = $root.tensorflow.NameAttrList.toObject(message.func, options);
                    if (options.oneofs)
                        object.value = "func";
                }
                return object;
            };
    
            AttrValue.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                ListValue.create = function create(properties) {
                    return new ListValue(properties);
                };
    
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
    
                ListValue.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.s != null && message.hasOwnProperty("s")) {
                        if (!Array.isArray(message.s))
                            return "s: array expected";
                        for (var i = 0; i < message.s.length; ++i)
                            if (!(message.s[i] && typeof message.s[i].length === "number" || $util.isString(message.s[i])))
                                return "s: buffer[] expected";
                    }
                    if (message.i != null && message.hasOwnProperty("i")) {
                        if (!Array.isArray(message.i))
                            return "i: array expected";
                        for (var i = 0; i < message.i.length; ++i)
                            if (!$util.isInteger(message.i[i]) && !(message.i[i] && $util.isInteger(message.i[i].low) && $util.isInteger(message.i[i].high)))
                                return "i: integer|Long[] expected";
                    }
                    if (message.f != null && message.hasOwnProperty("f")) {
                        if (!Array.isArray(message.f))
                            return "f: array expected";
                        for (var i = 0; i < message.f.length; ++i)
                            if (typeof message.f[i] !== "number")
                                return "f: number[] expected";
                    }
                    if (message.b != null && message.hasOwnProperty("b")) {
                        if (!Array.isArray(message.b))
                            return "b: array expected";
                        for (var i = 0; i < message.b.length; ++i)
                            if (typeof message.b[i] !== "boolean")
                                return "b: boolean[] expected";
                    }
                    if (message.type != null && message.hasOwnProperty("type")) {
                        if (!Array.isArray(message.type))
                            return "type: array expected";
                        for (var i = 0; i < message.type.length; ++i)
                            switch (message.type[i]) {
                            default:
                                return "type: enum value[] expected";
                            case 0:
                            case 1:
                            case 2:
                            case 3:
                            case 4:
                            case 5:
                            case 6:
                            case 7:
                            case 8:
                            case 9:
                            case 10:
                            case 11:
                            case 12:
                            case 13:
                            case 14:
                            case 15:
                            case 16:
                            case 17:
                            case 18:
                            case 19:
                            case 20:
                            case 21:
                            case 22:
                            case 23:
                            case 101:
                            case 102:
                            case 103:
                            case 104:
                            case 105:
                            case 106:
                            case 107:
                            case 108:
                            case 109:
                            case 110:
                            case 111:
                            case 112:
                            case 113:
                            case 114:
                            case 115:
                            case 116:
                            case 117:
                            case 118:
                            case 119:
                            case 120:
                            case 121:
                            case 122:
                            case 123:
                                break;
                            }
                    }
                    if (message.shape != null && message.hasOwnProperty("shape")) {
                        if (!Array.isArray(message.shape))
                            return "shape: array expected";
                        for (var i = 0; i < message.shape.length; ++i) {
                            var error = $root.tensorflow.TensorShapeProto.verify(message.shape[i]);
                            if (error)
                                return "shape." + error;
                        }
                    }
                    if (message.tensor != null && message.hasOwnProperty("tensor")) {
                        if (!Array.isArray(message.tensor))
                            return "tensor: array expected";
                        for (var i = 0; i < message.tensor.length; ++i) {
                            var error = $root.tensorflow.TensorProto.verify(message.tensor[i]);
                            if (error)
                                return "tensor." + error;
                        }
                    }
                    if (message.func != null && message.hasOwnProperty("func")) {
                        if (!Array.isArray(message.func))
                            return "func: array expected";
                        for (var i = 0; i < message.func.length; ++i) {
                            var error = $root.tensorflow.NameAttrList.verify(message.func[i]);
                            if (error)
                                return "func." + error;
                        }
                    }
                    return null;
                };
    
                ListValue.fromObject = function fromObject(object) {
                    if (object instanceof $root.tensorflow.AttrValue.ListValue)
                        return object;
                    var message = new $root.tensorflow.AttrValue.ListValue();
                    if (object.s) {
                        if (!Array.isArray(object.s))
                            throw TypeError(".tensorflow.AttrValue.ListValue.s: array expected");
                        message.s = [];
                        for (var i = 0; i < object.s.length; ++i)
                            if (typeof object.s[i] === "string")
                                $util.base64.decode(object.s[i], message.s[i] = $util.newBuffer($util.base64.length(object.s[i])), 0);
                            else if (object.s[i].length)
                                message.s[i] = object.s[i];
                    }
                    if (object.i) {
                        if (!Array.isArray(object.i))
                            throw TypeError(".tensorflow.AttrValue.ListValue.i: array expected");
                        message.i = [];
                        for (var i = 0; i < object.i.length; ++i)
                            if ($util.Long)
                                (message.i[i] = $util.Long.fromValue(object.i[i])).unsigned = false;
                            else if (typeof object.i[i] === "string")
                                message.i[i] = parseInt(object.i[i], 10);
                            else if (typeof object.i[i] === "number")
                                message.i[i] = object.i[i];
                            else if (typeof object.i[i] === "object")
                                message.i[i] = new $util.LongBits(object.i[i].low >>> 0, object.i[i].high >>> 0).toNumber();
                    }
                    if (object.f) {
                        if (!Array.isArray(object.f))
                            throw TypeError(".tensorflow.AttrValue.ListValue.f: array expected");
                        message.f = [];
                        for (var i = 0; i < object.f.length; ++i)
                            message.f[i] = Number(object.f[i]);
                    }
                    if (object.b) {
                        if (!Array.isArray(object.b))
                            throw TypeError(".tensorflow.AttrValue.ListValue.b: array expected");
                        message.b = [];
                        for (var i = 0; i < object.b.length; ++i)
                            message.b[i] = Boolean(object.b[i]);
                    }
                    if (object.type) {
                        if (!Array.isArray(object.type))
                            throw TypeError(".tensorflow.AttrValue.ListValue.type: array expected");
                        message.type = [];
                        for (var i = 0; i < object.type.length; ++i)
                            switch (object.type[i]) {
                            default:
                            case "DT_INVALID":
                            case 0:
                                message.type[i] = 0;
                                break;
                            case "DT_FLOAT":
                            case 1:
                                message.type[i] = 1;
                                break;
                            case "DT_DOUBLE":
                            case 2:
                                message.type[i] = 2;
                                break;
                            case "DT_INT32":
                            case 3:
                                message.type[i] = 3;
                                break;
                            case "DT_UINT8":
                            case 4:
                                message.type[i] = 4;
                                break;
                            case "DT_INT16":
                            case 5:
                                message.type[i] = 5;
                                break;
                            case "DT_INT8":
                            case 6:
                                message.type[i] = 6;
                                break;
                            case "DT_STRING":
                            case 7:
                                message.type[i] = 7;
                                break;
                            case "DT_COMPLEX64":
                            case 8:
                                message.type[i] = 8;
                                break;
                            case "DT_INT64":
                            case 9:
                                message.type[i] = 9;
                                break;
                            case "DT_BOOL":
                            case 10:
                                message.type[i] = 10;
                                break;
                            case "DT_QINT8":
                            case 11:
                                message.type[i] = 11;
                                break;
                            case "DT_QUINT8":
                            case 12:
                                message.type[i] = 12;
                                break;
                            case "DT_QINT32":
                            case 13:
                                message.type[i] = 13;
                                break;
                            case "DT_BFLOAT16":
                            case 14:
                                message.type[i] = 14;
                                break;
                            case "DT_QINT16":
                            case 15:
                                message.type[i] = 15;
                                break;
                            case "DT_QUINT16":
                            case 16:
                                message.type[i] = 16;
                                break;
                            case "DT_UINT16":
                            case 17:
                                message.type[i] = 17;
                                break;
                            case "DT_COMPLEX128":
                            case 18:
                                message.type[i] = 18;
                                break;
                            case "DT_HALF":
                            case 19:
                                message.type[i] = 19;
                                break;
                            case "DT_RESOURCE":
                            case 20:
                                message.type[i] = 20;
                                break;
                            case "DT_VARIANT":
                            case 21:
                                message.type[i] = 21;
                                break;
                            case "DT_UINT32":
                            case 22:
                                message.type[i] = 22;
                                break;
                            case "DT_UINT64":
                            case 23:
                                message.type[i] = 23;
                                break;
                            case "DT_FLOAT_REF":
                            case 101:
                                message.type[i] = 101;
                                break;
                            case "DT_DOUBLE_REF":
                            case 102:
                                message.type[i] = 102;
                                break;
                            case "DT_INT32_REF":
                            case 103:
                                message.type[i] = 103;
                                break;
                            case "DT_UINT8_REF":
                            case 104:
                                message.type[i] = 104;
                                break;
                            case "DT_INT16_REF":
                            case 105:
                                message.type[i] = 105;
                                break;
                            case "DT_INT8_REF":
                            case 106:
                                message.type[i] = 106;
                                break;
                            case "DT_STRING_REF":
                            case 107:
                                message.type[i] = 107;
                                break;
                            case "DT_COMPLEX64_REF":
                            case 108:
                                message.type[i] = 108;
                                break;
                            case "DT_INT64_REF":
                            case 109:
                                message.type[i] = 109;
                                break;
                            case "DT_BOOL_REF":
                            case 110:
                                message.type[i] = 110;
                                break;
                            case "DT_QINT8_REF":
                            case 111:
                                message.type[i] = 111;
                                break;
                            case "DT_QUINT8_REF":
                            case 112:
                                message.type[i] = 112;
                                break;
                            case "DT_QINT32_REF":
                            case 113:
                                message.type[i] = 113;
                                break;
                            case "DT_BFLOAT16_REF":
                            case 114:
                                message.type[i] = 114;
                                break;
                            case "DT_QINT16_REF":
                            case 115:
                                message.type[i] = 115;
                                break;
                            case "DT_QUINT16_REF":
                            case 116:
                                message.type[i] = 116;
                                break;
                            case "DT_UINT16_REF":
                            case 117:
                                message.type[i] = 117;
                                break;
                            case "DT_COMPLEX128_REF":
                            case 118:
                                message.type[i] = 118;
                                break;
                            case "DT_HALF_REF":
                            case 119:
                                message.type[i] = 119;
                                break;
                            case "DT_RESOURCE_REF":
                            case 120:
                                message.type[i] = 120;
                                break;
                            case "DT_VARIANT_REF":
                            case 121:
                                message.type[i] = 121;
                                break;
                            case "DT_UINT32_REF":
                            case 122:
                                message.type[i] = 122;
                                break;
                            case "DT_UINT64_REF":
                            case 123:
                                message.type[i] = 123;
                                break;
                            }
                    }
                    if (object.shape) {
                        if (!Array.isArray(object.shape))
                            throw TypeError(".tensorflow.AttrValue.ListValue.shape: array expected");
                        message.shape = [];
                        for (var i = 0; i < object.shape.length; ++i) {
                            if (typeof object.shape[i] !== "object")
                                throw TypeError(".tensorflow.AttrValue.ListValue.shape: object expected");
                            message.shape[i] = $root.tensorflow.TensorShapeProto.fromObject(object.shape[i]);
                        }
                    }
                    if (object.tensor) {
                        if (!Array.isArray(object.tensor))
                            throw TypeError(".tensorflow.AttrValue.ListValue.tensor: array expected");
                        message.tensor = [];
                        for (var i = 0; i < object.tensor.length; ++i) {
                            if (typeof object.tensor[i] !== "object")
                                throw TypeError(".tensorflow.AttrValue.ListValue.tensor: object expected");
                            message.tensor[i] = $root.tensorflow.TensorProto.fromObject(object.tensor[i]);
                        }
                    }
                    if (object.func) {
                        if (!Array.isArray(object.func))
                            throw TypeError(".tensorflow.AttrValue.ListValue.func: array expected");
                        message.func = [];
                        for (var i = 0; i < object.func.length; ++i) {
                            if (typeof object.func[i] !== "object")
                                throw TypeError(".tensorflow.AttrValue.ListValue.func: object expected");
                            message.func[i] = $root.tensorflow.NameAttrList.fromObject(object.func[i]);
                        }
                    }
                    return message;
                };
    
                ListValue.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.arrays || options.defaults) {
                        object.s = [];
                        object.i = [];
                        object.f = [];
                        object.b = [];
                        object.type = [];
                        object.shape = [];
                        object.tensor = [];
                        object.func = [];
                    }
                    if (message.s && message.s.length) {
                        object.s = [];
                        for (var j = 0; j < message.s.length; ++j)
                            object.s[j] = options.bytes === String ? $util.base64.encode(message.s[j], 0, message.s[j].length) : options.bytes === Array ? Array.prototype.slice.call(message.s[j]) : message.s[j];
                    }
                    if (message.i && message.i.length) {
                        object.i = [];
                        for (var j = 0; j < message.i.length; ++j)
                            if (typeof message.i[j] === "number")
                                object.i[j] = options.longs === String ? String(message.i[j]) : message.i[j];
                            else
                                object.i[j] = options.longs === String ? $util.Long.prototype.toString.call(message.i[j]) : options.longs === Number ? new $util.LongBits(message.i[j].low >>> 0, message.i[j].high >>> 0).toNumber() : message.i[j];
                    }
                    if (message.f && message.f.length) {
                        object.f = [];
                        for (var j = 0; j < message.f.length; ++j)
                            object.f[j] = options.json && !isFinite(message.f[j]) ? String(message.f[j]) : message.f[j];
                    }
                    if (message.b && message.b.length) {
                        object.b = [];
                        for (var j = 0; j < message.b.length; ++j)
                            object.b[j] = message.b[j];
                    }
                    if (message.type && message.type.length) {
                        object.type = [];
                        for (var j = 0; j < message.type.length; ++j)
                            object.type[j] = options.enums === String ? $root.tensorflow.DataType[message.type[j]] : message.type[j];
                    }
                    if (message.shape && message.shape.length) {
                        object.shape = [];
                        for (var j = 0; j < message.shape.length; ++j)
                            object.shape[j] = $root.tensorflow.TensorShapeProto.toObject(message.shape[j], options);
                    }
                    if (message.tensor && message.tensor.length) {
                        object.tensor = [];
                        for (var j = 0; j < message.tensor.length; ++j)
                            object.tensor[j] = $root.tensorflow.TensorProto.toObject(message.tensor[j], options);
                    }
                    if (message.func && message.func.length) {
                        object.func = [];
                        for (var j = 0; j < message.func.length; ++j)
                            object.func[j] = $root.tensorflow.NameAttrList.toObject(message.func[j], options);
                    }
                    return object;
                };
    
                ListValue.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
            NameAttrList.create = function create(properties) {
                return new NameAttrList(properties);
            };
    
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
    
            NameAttrList.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.attr != null && message.hasOwnProperty("attr")) {
                    if (!$util.isObject(message.attr))
                        return "attr: object expected";
                    var key = Object.keys(message.attr);
                    for (var i = 0; i < key.length; ++i) {
                        var error = $root.tensorflow.AttrValue.verify(message.attr[key[i]]);
                        if (error)
                            return "attr." + error;
                    }
                }
                return null;
            };
    
            NameAttrList.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.NameAttrList)
                    return object;
                var message = new $root.tensorflow.NameAttrList();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.attr) {
                    if (typeof object.attr !== "object")
                        throw TypeError(".tensorflow.NameAttrList.attr: object expected");
                    message.attr = {};
                    for (var keys = Object.keys(object.attr), i = 0; i < keys.length; ++i) {
                        if (typeof object.attr[keys[i]] !== "object")
                            throw TypeError(".tensorflow.NameAttrList.attr: object expected");
                        message.attr[keys[i]] = $root.tensorflow.AttrValue.fromObject(object.attr[keys[i]]);
                    }
                }
                return message;
            };
    
            NameAttrList.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.objects || options.defaults)
                    object.attr = {};
                if (options.defaults)
                    object.name = "";
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                var keys2;
                if (message.attr && (keys2 = Object.keys(message.attr)).length) {
                    object.attr = {};
                    for (var j = 0; j < keys2.length; ++j)
                        object.attr[keys2[j]] = $root.tensorflow.AttrValue.toObject(message.attr[keys2[j]], options);
                }
                return object;
            };
    
            NameAttrList.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return NameAttrList;
        })();
    
        tensorflow.TensorProto = (function() {
    
            function TensorProto(properties) {
                this.halfVal = [];
                this.floatVal = [];
                this.doubleVal = [];
                this.intVal = [];
                this.stringVal = [];
                this.scomplexVal = [];
                this.int64Val = [];
                this.boolVal = [];
                this.dcomplexVal = [];
                this.resourceHandleVal = [];
                this.variantVal = [];
                this.uint32Val = [];
                this.uint64Val = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorProto.prototype.dtype = 0;
            TensorProto.prototype.tensorShape = null;
            TensorProto.prototype.versionNumber = 0;
            TensorProto.prototype.tensorContent = $util.newBuffer([]);
            TensorProto.prototype.halfVal = $util.emptyArray;
            TensorProto.prototype.floatVal = $util.emptyArray;
            TensorProto.prototype.doubleVal = $util.emptyArray;
            TensorProto.prototype.intVal = $util.emptyArray;
            TensorProto.prototype.stringVal = $util.emptyArray;
            TensorProto.prototype.scomplexVal = $util.emptyArray;
            TensorProto.prototype.int64Val = $util.emptyArray;
            TensorProto.prototype.boolVal = $util.emptyArray;
            TensorProto.prototype.dcomplexVal = $util.emptyArray;
            TensorProto.prototype.resourceHandleVal = $util.emptyArray;
            TensorProto.prototype.variantVal = $util.emptyArray;
            TensorProto.prototype.uint32Val = $util.emptyArray;
            TensorProto.prototype.uint64Val = $util.emptyArray;
    
            TensorProto.create = function create(properties) {
                return new TensorProto(properties);
            };
    
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
                        message.tensorShape = $root.tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                        break;
                    case 3:
                        message.versionNumber = reader.int32();
                        break;
                    case 4:
                        message.tensorContent = reader.bytes();
                        break;
                    case 13:
                        if (!(message.halfVal && message.halfVal.length))
                            message.halfVal = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.halfVal.push(reader.int32());
                        } else
                            message.halfVal.push(reader.int32());
                        break;
                    case 5:
                        if (!(message.floatVal && message.floatVal.length))
                            message.floatVal = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.floatVal.push(reader.float());
                        } else
                            message.floatVal.push(reader.float());
                        break;
                    case 6:
                        if (!(message.doubleVal && message.doubleVal.length))
                            message.doubleVal = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.doubleVal.push(reader.double());
                        } else
                            message.doubleVal.push(reader.double());
                        break;
                    case 7:
                        if (!(message.intVal && message.intVal.length))
                            message.intVal = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.intVal.push(reader.int32());
                        } else
                            message.intVal.push(reader.int32());
                        break;
                    case 8:
                        if (!(message.stringVal && message.stringVal.length))
                            message.stringVal = [];
                        message.stringVal.push(reader.bytes());
                        break;
                    case 9:
                        if (!(message.scomplexVal && message.scomplexVal.length))
                            message.scomplexVal = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.scomplexVal.push(reader.float());
                        } else
                            message.scomplexVal.push(reader.float());
                        break;
                    case 10:
                        if (!(message.int64Val && message.int64Val.length))
                            message.int64Val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.int64Val.push(reader.int64());
                        } else
                            message.int64Val.push(reader.int64());
                        break;
                    case 11:
                        if (!(message.boolVal && message.boolVal.length))
                            message.boolVal = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.boolVal.push(reader.bool());
                        } else
                            message.boolVal.push(reader.bool());
                        break;
                    case 12:
                        if (!(message.dcomplexVal && message.dcomplexVal.length))
                            message.dcomplexVal = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.dcomplexVal.push(reader.double());
                        } else
                            message.dcomplexVal.push(reader.double());
                        break;
                    case 14:
                        if (!(message.resourceHandleVal && message.resourceHandleVal.length))
                            message.resourceHandleVal = [];
                        message.resourceHandleVal.push($root.tensorflow.ResourceHandleProto.decode(reader, reader.uint32()));
                        break;
                    case 15:
                        if (!(message.variantVal && message.variantVal.length))
                            message.variantVal = [];
                        message.variantVal.push($root.tensorflow.VariantTensorDataProto.decode(reader, reader.uint32()));
                        break;
                    case 16:
                        if (!(message.uint32Val && message.uint32Val.length))
                            message.uint32Val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.uint32Val.push(reader.uint32());
                        } else
                            message.uint32Val.push(reader.uint32());
                        break;
                    case 17:
                        if (!(message.uint64Val && message.uint64Val.length))
                            message.uint64Val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.uint64Val.push(reader.uint64());
                        } else
                            message.uint64Val.push(reader.uint64());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TensorProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.dtype != null && message.hasOwnProperty("dtype"))
                    switch (message.dtype) {
                    default:
                        return "dtype: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                    case 3:
                    case 4:
                    case 5:
                    case 6:
                    case 7:
                    case 8:
                    case 9:
                    case 10:
                    case 11:
                    case 12:
                    case 13:
                    case 14:
                    case 15:
                    case 16:
                    case 17:
                    case 18:
                    case 19:
                    case 20:
                    case 21:
                    case 22:
                    case 23:
                    case 101:
                    case 102:
                    case 103:
                    case 104:
                    case 105:
                    case 106:
                    case 107:
                    case 108:
                    case 109:
                    case 110:
                    case 111:
                    case 112:
                    case 113:
                    case 114:
                    case 115:
                    case 116:
                    case 117:
                    case 118:
                    case 119:
                    case 120:
                    case 121:
                    case 122:
                    case 123:
                        break;
                    }
                if (message.tensorShape != null && message.hasOwnProperty("tensorShape")) {
                    var error = $root.tensorflow.TensorShapeProto.verify(message.tensorShape);
                    if (error)
                        return "tensorShape." + error;
                }
                if (message.versionNumber != null && message.hasOwnProperty("versionNumber"))
                    if (!$util.isInteger(message.versionNumber))
                        return "versionNumber: integer expected";
                if (message.tensorContent != null && message.hasOwnProperty("tensorContent"))
                    if (!(message.tensorContent && typeof message.tensorContent.length === "number" || $util.isString(message.tensorContent)))
                        return "tensorContent: buffer expected";
                if (message.halfVal != null && message.hasOwnProperty("halfVal")) {
                    if (!Array.isArray(message.halfVal))
                        return "halfVal: array expected";
                    for (var i = 0; i < message.halfVal.length; ++i)
                        if (!$util.isInteger(message.halfVal[i]))
                            return "halfVal: integer[] expected";
                }
                if (message.floatVal != null && message.hasOwnProperty("floatVal")) {
                    if (!Array.isArray(message.floatVal))
                        return "floatVal: array expected";
                    for (var i = 0; i < message.floatVal.length; ++i)
                        if (typeof message.floatVal[i] !== "number")
                            return "floatVal: number[] expected";
                }
                if (message.doubleVal != null && message.hasOwnProperty("doubleVal")) {
                    if (!Array.isArray(message.doubleVal))
                        return "doubleVal: array expected";
                    for (var i = 0; i < message.doubleVal.length; ++i)
                        if (typeof message.doubleVal[i] !== "number")
                            return "doubleVal: number[] expected";
                }
                if (message.intVal != null && message.hasOwnProperty("intVal")) {
                    if (!Array.isArray(message.intVal))
                        return "intVal: array expected";
                    for (var i = 0; i < message.intVal.length; ++i)
                        if (!$util.isInteger(message.intVal[i]))
                            return "intVal: integer[] expected";
                }
                if (message.stringVal != null && message.hasOwnProperty("stringVal")) {
                    if (!Array.isArray(message.stringVal))
                        return "stringVal: array expected";
                    for (var i = 0; i < message.stringVal.length; ++i)
                        if (!(message.stringVal[i] && typeof message.stringVal[i].length === "number" || $util.isString(message.stringVal[i])))
                            return "stringVal: buffer[] expected";
                }
                if (message.scomplexVal != null && message.hasOwnProperty("scomplexVal")) {
                    if (!Array.isArray(message.scomplexVal))
                        return "scomplexVal: array expected";
                    for (var i = 0; i < message.scomplexVal.length; ++i)
                        if (typeof message.scomplexVal[i] !== "number")
                            return "scomplexVal: number[] expected";
                }
                if (message.int64Val != null && message.hasOwnProperty("int64Val")) {
                    if (!Array.isArray(message.int64Val))
                        return "int64Val: array expected";
                    for (var i = 0; i < message.int64Val.length; ++i)
                        if (!$util.isInteger(message.int64Val[i]) && !(message.int64Val[i] && $util.isInteger(message.int64Val[i].low) && $util.isInteger(message.int64Val[i].high)))
                            return "int64Val: integer|Long[] expected";
                }
                if (message.boolVal != null && message.hasOwnProperty("boolVal")) {
                    if (!Array.isArray(message.boolVal))
                        return "boolVal: array expected";
                    for (var i = 0; i < message.boolVal.length; ++i)
                        if (typeof message.boolVal[i] !== "boolean")
                            return "boolVal: boolean[] expected";
                }
                if (message.dcomplexVal != null && message.hasOwnProperty("dcomplexVal")) {
                    if (!Array.isArray(message.dcomplexVal))
                        return "dcomplexVal: array expected";
                    for (var i = 0; i < message.dcomplexVal.length; ++i)
                        if (typeof message.dcomplexVal[i] !== "number")
                            return "dcomplexVal: number[] expected";
                }
                if (message.resourceHandleVal != null && message.hasOwnProperty("resourceHandleVal")) {
                    if (!Array.isArray(message.resourceHandleVal))
                        return "resourceHandleVal: array expected";
                    for (var i = 0; i < message.resourceHandleVal.length; ++i) {
                        var error = $root.tensorflow.ResourceHandleProto.verify(message.resourceHandleVal[i]);
                        if (error)
                            return "resourceHandleVal." + error;
                    }
                }
                if (message.variantVal != null && message.hasOwnProperty("variantVal")) {
                    if (!Array.isArray(message.variantVal))
                        return "variantVal: array expected";
                    for (var i = 0; i < message.variantVal.length; ++i) {
                        var error = $root.tensorflow.VariantTensorDataProto.verify(message.variantVal[i]);
                        if (error)
                            return "variantVal." + error;
                    }
                }
                if (message.uint32Val != null && message.hasOwnProperty("uint32Val")) {
                    if (!Array.isArray(message.uint32Val))
                        return "uint32Val: array expected";
                    for (var i = 0; i < message.uint32Val.length; ++i)
                        if (!$util.isInteger(message.uint32Val[i]))
                            return "uint32Val: integer[] expected";
                }
                if (message.uint64Val != null && message.hasOwnProperty("uint64Val")) {
                    if (!Array.isArray(message.uint64Val))
                        return "uint64Val: array expected";
                    for (var i = 0; i < message.uint64Val.length; ++i)
                        if (!$util.isInteger(message.uint64Val[i]) && !(message.uint64Val[i] && $util.isInteger(message.uint64Val[i].low) && $util.isInteger(message.uint64Val[i].high)))
                            return "uint64Val: integer|Long[] expected";
                }
                return null;
            };
    
            TensorProto.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.TensorProto)
                    return object;
                var message = new $root.tensorflow.TensorProto();
                switch (object.dtype) {
                case "DT_INVALID":
                case 0:
                    message.dtype = 0;
                    break;
                case "DT_FLOAT":
                case 1:
                    message.dtype = 1;
                    break;
                case "DT_DOUBLE":
                case 2:
                    message.dtype = 2;
                    break;
                case "DT_INT32":
                case 3:
                    message.dtype = 3;
                    break;
                case "DT_UINT8":
                case 4:
                    message.dtype = 4;
                    break;
                case "DT_INT16":
                case 5:
                    message.dtype = 5;
                    break;
                case "DT_INT8":
                case 6:
                    message.dtype = 6;
                    break;
                case "DT_STRING":
                case 7:
                    message.dtype = 7;
                    break;
                case "DT_COMPLEX64":
                case 8:
                    message.dtype = 8;
                    break;
                case "DT_INT64":
                case 9:
                    message.dtype = 9;
                    break;
                case "DT_BOOL":
                case 10:
                    message.dtype = 10;
                    break;
                case "DT_QINT8":
                case 11:
                    message.dtype = 11;
                    break;
                case "DT_QUINT8":
                case 12:
                    message.dtype = 12;
                    break;
                case "DT_QINT32":
                case 13:
                    message.dtype = 13;
                    break;
                case "DT_BFLOAT16":
                case 14:
                    message.dtype = 14;
                    break;
                case "DT_QINT16":
                case 15:
                    message.dtype = 15;
                    break;
                case "DT_QUINT16":
                case 16:
                    message.dtype = 16;
                    break;
                case "DT_UINT16":
                case 17:
                    message.dtype = 17;
                    break;
                case "DT_COMPLEX128":
                case 18:
                    message.dtype = 18;
                    break;
                case "DT_HALF":
                case 19:
                    message.dtype = 19;
                    break;
                case "DT_RESOURCE":
                case 20:
                    message.dtype = 20;
                    break;
                case "DT_VARIANT":
                case 21:
                    message.dtype = 21;
                    break;
                case "DT_UINT32":
                case 22:
                    message.dtype = 22;
                    break;
                case "DT_UINT64":
                case 23:
                    message.dtype = 23;
                    break;
                case "DT_FLOAT_REF":
                case 101:
                    message.dtype = 101;
                    break;
                case "DT_DOUBLE_REF":
                case 102:
                    message.dtype = 102;
                    break;
                case "DT_INT32_REF":
                case 103:
                    message.dtype = 103;
                    break;
                case "DT_UINT8_REF":
                case 104:
                    message.dtype = 104;
                    break;
                case "DT_INT16_REF":
                case 105:
                    message.dtype = 105;
                    break;
                case "DT_INT8_REF":
                case 106:
                    message.dtype = 106;
                    break;
                case "DT_STRING_REF":
                case 107:
                    message.dtype = 107;
                    break;
                case "DT_COMPLEX64_REF":
                case 108:
                    message.dtype = 108;
                    break;
                case "DT_INT64_REF":
                case 109:
                    message.dtype = 109;
                    break;
                case "DT_BOOL_REF":
                case 110:
                    message.dtype = 110;
                    break;
                case "DT_QINT8_REF":
                case 111:
                    message.dtype = 111;
                    break;
                case "DT_QUINT8_REF":
                case 112:
                    message.dtype = 112;
                    break;
                case "DT_QINT32_REF":
                case 113:
                    message.dtype = 113;
                    break;
                case "DT_BFLOAT16_REF":
                case 114:
                    message.dtype = 114;
                    break;
                case "DT_QINT16_REF":
                case 115:
                    message.dtype = 115;
                    break;
                case "DT_QUINT16_REF":
                case 116:
                    message.dtype = 116;
                    break;
                case "DT_UINT16_REF":
                case 117:
                    message.dtype = 117;
                    break;
                case "DT_COMPLEX128_REF":
                case 118:
                    message.dtype = 118;
                    break;
                case "DT_HALF_REF":
                case 119:
                    message.dtype = 119;
                    break;
                case "DT_RESOURCE_REF":
                case 120:
                    message.dtype = 120;
                    break;
                case "DT_VARIANT_REF":
                case 121:
                    message.dtype = 121;
                    break;
                case "DT_UINT32_REF":
                case 122:
                    message.dtype = 122;
                    break;
                case "DT_UINT64_REF":
                case 123:
                    message.dtype = 123;
                    break;
                }
                if (object.tensorShape != null) {
                    if (typeof object.tensorShape !== "object")
                        throw TypeError(".tensorflow.TensorProto.tensorShape: object expected");
                    message.tensorShape = $root.tensorflow.TensorShapeProto.fromObject(object.tensorShape);
                }
                if (object.versionNumber != null)
                    message.versionNumber = object.versionNumber | 0;
                if (object.tensorContent != null)
                    if (typeof object.tensorContent === "string")
                        $util.base64.decode(object.tensorContent, message.tensorContent = $util.newBuffer($util.base64.length(object.tensorContent)), 0);
                    else if (object.tensorContent.length)
                        message.tensorContent = object.tensorContent;
                if (object.halfVal) {
                    if (!Array.isArray(object.halfVal))
                        throw TypeError(".tensorflow.TensorProto.halfVal: array expected");
                    message.halfVal = [];
                    for (var i = 0; i < object.halfVal.length; ++i)
                        message.halfVal[i] = object.halfVal[i] | 0;
                }
                if (object.floatVal) {
                    if (!Array.isArray(object.floatVal))
                        throw TypeError(".tensorflow.TensorProto.floatVal: array expected");
                    message.floatVal = [];
                    for (var i = 0; i < object.floatVal.length; ++i)
                        message.floatVal[i] = Number(object.floatVal[i]);
                }
                if (object.doubleVal) {
                    if (!Array.isArray(object.doubleVal))
                        throw TypeError(".tensorflow.TensorProto.doubleVal: array expected");
                    message.doubleVal = [];
                    for (var i = 0; i < object.doubleVal.length; ++i)
                        message.doubleVal[i] = Number(object.doubleVal[i]);
                }
                if (object.intVal) {
                    if (!Array.isArray(object.intVal))
                        throw TypeError(".tensorflow.TensorProto.intVal: array expected");
                    message.intVal = [];
                    for (var i = 0; i < object.intVal.length; ++i)
                        message.intVal[i] = object.intVal[i] | 0;
                }
                if (object.stringVal) {
                    if (!Array.isArray(object.stringVal))
                        throw TypeError(".tensorflow.TensorProto.stringVal: array expected");
                    message.stringVal = [];
                    for (var i = 0; i < object.stringVal.length; ++i)
                        if (typeof object.stringVal[i] === "string")
                            $util.base64.decode(object.stringVal[i], message.stringVal[i] = $util.newBuffer($util.base64.length(object.stringVal[i])), 0);
                        else if (object.stringVal[i].length)
                            message.stringVal[i] = object.stringVal[i];
                }
                if (object.scomplexVal) {
                    if (!Array.isArray(object.scomplexVal))
                        throw TypeError(".tensorflow.TensorProto.scomplexVal: array expected");
                    message.scomplexVal = [];
                    for (var i = 0; i < object.scomplexVal.length; ++i)
                        message.scomplexVal[i] = Number(object.scomplexVal[i]);
                }
                if (object.int64Val) {
                    if (!Array.isArray(object.int64Val))
                        throw TypeError(".tensorflow.TensorProto.int64Val: array expected");
                    message.int64Val = [];
                    for (var i = 0; i < object.int64Val.length; ++i)
                        if ($util.Long)
                            (message.int64Val[i] = $util.Long.fromValue(object.int64Val[i])).unsigned = false;
                        else if (typeof object.int64Val[i] === "string")
                            message.int64Val[i] = parseInt(object.int64Val[i], 10);
                        else if (typeof object.int64Val[i] === "number")
                            message.int64Val[i] = object.int64Val[i];
                        else if (typeof object.int64Val[i] === "object")
                            message.int64Val[i] = new $util.LongBits(object.int64Val[i].low >>> 0, object.int64Val[i].high >>> 0).toNumber();
                }
                if (object.boolVal) {
                    if (!Array.isArray(object.boolVal))
                        throw TypeError(".tensorflow.TensorProto.boolVal: array expected");
                    message.boolVal = [];
                    for (var i = 0; i < object.boolVal.length; ++i)
                        message.boolVal[i] = Boolean(object.boolVal[i]);
                }
                if (object.dcomplexVal) {
                    if (!Array.isArray(object.dcomplexVal))
                        throw TypeError(".tensorflow.TensorProto.dcomplexVal: array expected");
                    message.dcomplexVal = [];
                    for (var i = 0; i < object.dcomplexVal.length; ++i)
                        message.dcomplexVal[i] = Number(object.dcomplexVal[i]);
                }
                if (object.resourceHandleVal) {
                    if (!Array.isArray(object.resourceHandleVal))
                        throw TypeError(".tensorflow.TensorProto.resourceHandleVal: array expected");
                    message.resourceHandleVal = [];
                    for (var i = 0; i < object.resourceHandleVal.length; ++i) {
                        if (typeof object.resourceHandleVal[i] !== "object")
                            throw TypeError(".tensorflow.TensorProto.resourceHandleVal: object expected");
                        message.resourceHandleVal[i] = $root.tensorflow.ResourceHandleProto.fromObject(object.resourceHandleVal[i]);
                    }
                }
                if (object.variantVal) {
                    if (!Array.isArray(object.variantVal))
                        throw TypeError(".tensorflow.TensorProto.variantVal: array expected");
                    message.variantVal = [];
                    for (var i = 0; i < object.variantVal.length; ++i) {
                        if (typeof object.variantVal[i] !== "object")
                            throw TypeError(".tensorflow.TensorProto.variantVal: object expected");
                        message.variantVal[i] = $root.tensorflow.VariantTensorDataProto.fromObject(object.variantVal[i]);
                    }
                }
                if (object.uint32Val) {
                    if (!Array.isArray(object.uint32Val))
                        throw TypeError(".tensorflow.TensorProto.uint32Val: array expected");
                    message.uint32Val = [];
                    for (var i = 0; i < object.uint32Val.length; ++i)
                        message.uint32Val[i] = object.uint32Val[i] >>> 0;
                }
                if (object.uint64Val) {
                    if (!Array.isArray(object.uint64Val))
                        throw TypeError(".tensorflow.TensorProto.uint64Val: array expected");
                    message.uint64Val = [];
                    for (var i = 0; i < object.uint64Val.length; ++i)
                        if ($util.Long)
                            (message.uint64Val[i] = $util.Long.fromValue(object.uint64Val[i])).unsigned = true;
                        else if (typeof object.uint64Val[i] === "string")
                            message.uint64Val[i] = parseInt(object.uint64Val[i], 10);
                        else if (typeof object.uint64Val[i] === "number")
                            message.uint64Val[i] = object.uint64Val[i];
                        else if (typeof object.uint64Val[i] === "object")
                            message.uint64Val[i] = new $util.LongBits(object.uint64Val[i].low >>> 0, object.uint64Val[i].high >>> 0).toNumber(true);
                }
                return message;
            };
    
            TensorProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.floatVal = [];
                    object.doubleVal = [];
                    object.intVal = [];
                    object.stringVal = [];
                    object.scomplexVal = [];
                    object.int64Val = [];
                    object.boolVal = [];
                    object.dcomplexVal = [];
                    object.halfVal = [];
                    object.resourceHandleVal = [];
                    object.variantVal = [];
                    object.uint32Val = [];
                    object.uint64Val = [];
                }
                if (options.defaults) {
                    object.dtype = options.enums === String ? "DT_INVALID" : 0;
                    object.tensorShape = null;
                    object.versionNumber = 0;
                    if (options.bytes === String)
                        object.tensorContent = "";
                    else {
                        object.tensorContent = [];
                        if (options.bytes !== Array)
                            object.tensorContent = $util.newBuffer(object.tensorContent);
                    }
                }
                if (message.dtype != null && message.hasOwnProperty("dtype"))
                    object.dtype = options.enums === String ? $root.tensorflow.DataType[message.dtype] : message.dtype;
                if (message.tensorShape != null && message.hasOwnProperty("tensorShape"))
                    object.tensorShape = $root.tensorflow.TensorShapeProto.toObject(message.tensorShape, options);
                if (message.versionNumber != null && message.hasOwnProperty("versionNumber"))
                    object.versionNumber = message.versionNumber;
                if (message.tensorContent != null && message.hasOwnProperty("tensorContent"))
                    object.tensorContent = options.bytes === String ? $util.base64.encode(message.tensorContent, 0, message.tensorContent.length) : options.bytes === Array ? Array.prototype.slice.call(message.tensorContent) : message.tensorContent;
                if (message.floatVal && message.floatVal.length) {
                    object.floatVal = [];
                    for (var j = 0; j < message.floatVal.length; ++j)
                        object.floatVal[j] = options.json && !isFinite(message.floatVal[j]) ? String(message.floatVal[j]) : message.floatVal[j];
                }
                if (message.doubleVal && message.doubleVal.length) {
                    object.doubleVal = [];
                    for (var j = 0; j < message.doubleVal.length; ++j)
                        object.doubleVal[j] = options.json && !isFinite(message.doubleVal[j]) ? String(message.doubleVal[j]) : message.doubleVal[j];
                }
                if (message.intVal && message.intVal.length) {
                    object.intVal = [];
                    for (var j = 0; j < message.intVal.length; ++j)
                        object.intVal[j] = message.intVal[j];
                }
                if (message.stringVal && message.stringVal.length) {
                    object.stringVal = [];
                    for (var j = 0; j < message.stringVal.length; ++j)
                        object.stringVal[j] = options.bytes === String ? $util.base64.encode(message.stringVal[j], 0, message.stringVal[j].length) : options.bytes === Array ? Array.prototype.slice.call(message.stringVal[j]) : message.stringVal[j];
                }
                if (message.scomplexVal && message.scomplexVal.length) {
                    object.scomplexVal = [];
                    for (var j = 0; j < message.scomplexVal.length; ++j)
                        object.scomplexVal[j] = options.json && !isFinite(message.scomplexVal[j]) ? String(message.scomplexVal[j]) : message.scomplexVal[j];
                }
                if (message.int64Val && message.int64Val.length) {
                    object.int64Val = [];
                    for (var j = 0; j < message.int64Val.length; ++j)
                        if (typeof message.int64Val[j] === "number")
                            object.int64Val[j] = options.longs === String ? String(message.int64Val[j]) : message.int64Val[j];
                        else
                            object.int64Val[j] = options.longs === String ? $util.Long.prototype.toString.call(message.int64Val[j]) : options.longs === Number ? new $util.LongBits(message.int64Val[j].low >>> 0, message.int64Val[j].high >>> 0).toNumber() : message.int64Val[j];
                }
                if (message.boolVal && message.boolVal.length) {
                    object.boolVal = [];
                    for (var j = 0; j < message.boolVal.length; ++j)
                        object.boolVal[j] = message.boolVal[j];
                }
                if (message.dcomplexVal && message.dcomplexVal.length) {
                    object.dcomplexVal = [];
                    for (var j = 0; j < message.dcomplexVal.length; ++j)
                        object.dcomplexVal[j] = options.json && !isFinite(message.dcomplexVal[j]) ? String(message.dcomplexVal[j]) : message.dcomplexVal[j];
                }
                if (message.halfVal && message.halfVal.length) {
                    object.halfVal = [];
                    for (var j = 0; j < message.halfVal.length; ++j)
                        object.halfVal[j] = message.halfVal[j];
                }
                if (message.resourceHandleVal && message.resourceHandleVal.length) {
                    object.resourceHandleVal = [];
                    for (var j = 0; j < message.resourceHandleVal.length; ++j)
                        object.resourceHandleVal[j] = $root.tensorflow.ResourceHandleProto.toObject(message.resourceHandleVal[j], options);
                }
                if (message.variantVal && message.variantVal.length) {
                    object.variantVal = [];
                    for (var j = 0; j < message.variantVal.length; ++j)
                        object.variantVal[j] = $root.tensorflow.VariantTensorDataProto.toObject(message.variantVal[j], options);
                }
                if (message.uint32Val && message.uint32Val.length) {
                    object.uint32Val = [];
                    for (var j = 0; j < message.uint32Val.length; ++j)
                        object.uint32Val[j] = message.uint32Val[j];
                }
                if (message.uint64Val && message.uint64Val.length) {
                    object.uint64Val = [];
                    for (var j = 0; j < message.uint64Val.length; ++j)
                        if (typeof message.uint64Val[j] === "number")
                            object.uint64Val[j] = options.longs === String ? String(message.uint64Val[j]) : message.uint64Val[j];
                        else
                            object.uint64Val[j] = options.longs === String ? $util.Long.prototype.toString.call(message.uint64Val[j]) : options.longs === Number ? new $util.LongBits(message.uint64Val[j].low >>> 0, message.uint64Val[j].high >>> 0).toNumber(true) : message.uint64Val[j];
                }
                return object;
            };
    
            TensorProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
            VariantTensorDataProto.prototype.typeName = "";
            VariantTensorDataProto.prototype.metadata = $util.newBuffer([]);
            VariantTensorDataProto.prototype.tensors = $util.emptyArray;
    
            VariantTensorDataProto.create = function create(properties) {
                return new VariantTensorDataProto(properties);
            };
    
            VariantTensorDataProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.tensorflow.VariantTensorDataProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.typeName = reader.string();
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
    
            VariantTensorDataProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.typeName != null && message.hasOwnProperty("typeName"))
                    if (!$util.isString(message.typeName))
                        return "typeName: string expected";
                if (message.metadata != null && message.hasOwnProperty("metadata"))
                    if (!(message.metadata && typeof message.metadata.length === "number" || $util.isString(message.metadata)))
                        return "metadata: buffer expected";
                if (message.tensors != null && message.hasOwnProperty("tensors")) {
                    if (!Array.isArray(message.tensors))
                        return "tensors: array expected";
                    for (var i = 0; i < message.tensors.length; ++i) {
                        var error = $root.tensorflow.TensorProto.verify(message.tensors[i]);
                        if (error)
                            return "tensors." + error;
                    }
                }
                return null;
            };
    
            VariantTensorDataProto.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.VariantTensorDataProto)
                    return object;
                var message = new $root.tensorflow.VariantTensorDataProto();
                if (object.typeName != null)
                    message.typeName = String(object.typeName);
                if (object.metadata != null)
                    if (typeof object.metadata === "string")
                        $util.base64.decode(object.metadata, message.metadata = $util.newBuffer($util.base64.length(object.metadata)), 0);
                    else if (object.metadata.length)
                        message.metadata = object.metadata;
                if (object.tensors) {
                    if (!Array.isArray(object.tensors))
                        throw TypeError(".tensorflow.VariantTensorDataProto.tensors: array expected");
                    message.tensors = [];
                    for (var i = 0; i < object.tensors.length; ++i) {
                        if (typeof object.tensors[i] !== "object")
                            throw TypeError(".tensorflow.VariantTensorDataProto.tensors: object expected");
                        message.tensors[i] = $root.tensorflow.TensorProto.fromObject(object.tensors[i]);
                    }
                }
                return message;
            };
    
            VariantTensorDataProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.tensors = [];
                if (options.defaults) {
                    object.typeName = "";
                    if (options.bytes === String)
                        object.metadata = "";
                    else {
                        object.metadata = [];
                        if (options.bytes !== Array)
                            object.metadata = $util.newBuffer(object.metadata);
                    }
                }
                if (message.typeName != null && message.hasOwnProperty("typeName"))
                    object.typeName = message.typeName;
                if (message.metadata != null && message.hasOwnProperty("metadata"))
                    object.metadata = options.bytes === String ? $util.base64.encode(message.metadata, 0, message.metadata.length) : options.bytes === Array ? Array.prototype.slice.call(message.metadata) : message.metadata;
                if (message.tensors && message.tensors.length) {
                    object.tensors = [];
                    for (var j = 0; j < message.tensors.length; ++j)
                        object.tensors[j] = $root.tensorflow.TensorProto.toObject(message.tensors[j], options);
                }
                return object;
            };
    
            VariantTensorDataProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return VariantTensorDataProto;
        })();
    
        tensorflow.ResourceHandleProto = (function() {
    
            function ResourceHandleProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ResourceHandleProto.prototype.device = "";
            ResourceHandleProto.prototype.container = "";
            ResourceHandleProto.prototype.name = "";
            ResourceHandleProto.prototype.hashCode = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
            ResourceHandleProto.prototype.maybeTypeName = "";
    
            ResourceHandleProto.create = function create(properties) {
                return new ResourceHandleProto(properties);
            };
    
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
                        message.hashCode = reader.uint64();
                        break;
                    case 5:
                        message.maybeTypeName = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ResourceHandleProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.device != null && message.hasOwnProperty("device"))
                    if (!$util.isString(message.device))
                        return "device: string expected";
                if (message.container != null && message.hasOwnProperty("container"))
                    if (!$util.isString(message.container))
                        return "container: string expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.hashCode != null && message.hasOwnProperty("hashCode"))
                    if (!$util.isInteger(message.hashCode) && !(message.hashCode && $util.isInteger(message.hashCode.low) && $util.isInteger(message.hashCode.high)))
                        return "hashCode: integer|Long expected";
                if (message.maybeTypeName != null && message.hasOwnProperty("maybeTypeName"))
                    if (!$util.isString(message.maybeTypeName))
                        return "maybeTypeName: string expected";
                return null;
            };
    
            ResourceHandleProto.fromObject = function fromObject(object) {
                if (object instanceof $root.tensorflow.ResourceHandleProto)
                    return object;
                var message = new $root.tensorflow.ResourceHandleProto();
                if (object.device != null)
                    message.device = String(object.device);
                if (object.container != null)
                    message.container = String(object.container);
                if (object.name != null)
                    message.name = String(object.name);
                if (object.hashCode != null)
                    if ($util.Long)
                        (message.hashCode = $util.Long.fromValue(object.hashCode)).unsigned = true;
                    else if (typeof object.hashCode === "string")
                        message.hashCode = parseInt(object.hashCode, 10);
                    else if (typeof object.hashCode === "number")
                        message.hashCode = object.hashCode;
                    else if (typeof object.hashCode === "object")
                        message.hashCode = new $util.LongBits(object.hashCode.low >>> 0, object.hashCode.high >>> 0).toNumber(true);
                if (object.maybeTypeName != null)
                    message.maybeTypeName = String(object.maybeTypeName);
                return message;
            };
    
            ResourceHandleProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.device = "";
                    object.container = "";
                    object.name = "";
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, true);
                        object.hashCode = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.hashCode = options.longs === String ? "0" : 0;
                    object.maybeTypeName = "";
                }
                if (message.device != null && message.hasOwnProperty("device"))
                    object.device = message.device;
                if (message.container != null && message.hasOwnProperty("container"))
                    object.container = message.container;
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.hashCode != null && message.hasOwnProperty("hashCode"))
                    if (typeof message.hashCode === "number")
                        object.hashCode = options.longs === String ? String(message.hashCode) : message.hashCode;
                    else
                        object.hashCode = options.longs === String ? $util.Long.prototype.toString.call(message.hashCode) : options.longs === Number ? new $util.LongBits(message.hashCode.low >>> 0, message.hashCode.high >>> 0).toNumber(true) : message.hashCode;
                if (message.maybeTypeName != null && message.hasOwnProperty("maybeTypeName"))
                    object.maybeTypeName = message.maybeTypeName;
                return object;
            };
    
            ResourceHandleProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ResourceHandleProto;
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
    
                Any.create = function create(properties) {
                    return new Any(properties);
                };
    
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
    
                Any.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.type_url != null && message.hasOwnProperty("type_url"))
                        if (!$util.isString(message.type_url))
                            return "type_url: string expected";
                    if (message.value != null && message.hasOwnProperty("value"))
                        if (!(message.value && typeof message.value.length === "number" || $util.isString(message.value)))
                            return "value: buffer expected";
                    return null;
                };
    
                Any.fromObject = function fromObject(object) {
                    if (object instanceof $root.google.protobuf.Any)
                        return object;
                    var message = new $root.google.protobuf.Any();
                    if (object.type_url != null)
                        message.type_url = String(object.type_url);
                    if (object.value != null)
                        if (typeof object.value === "string")
                            $util.base64.decode(object.value, message.value = $util.newBuffer($util.base64.length(object.value)), 0);
                        else if (object.value.length)
                            message.value = object.value;
                    return message;
                };
    
                Any.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        object.type_url = "";
                        if (options.bytes === String)
                            object.value = "";
                        else {
                            object.value = [];
                            if (options.bytes !== Array)
                                object.value = $util.newBuffer(object.value);
                        }
                    }
                    if (message.type_url != null && message.hasOwnProperty("type_url"))
                        object.type_url = message.type_url;
                    if (message.value != null && message.hasOwnProperty("value"))
                        object.value = options.bytes === String ? $util.base64.encode(message.value, 0, message.value.length) : options.bytes === Array ? Array.prototype.slice.call(message.value) : message.value;
                    return object;
                };
    
                Any.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return Any;
            })();
    
            return protobuf;
        })();
    
        return google;
    })();

    return $root;
})(protobuf);
