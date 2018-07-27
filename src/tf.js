/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    // Common aliases
    var $Reader = $protobuf.Reader, $Writer = $protobuf.Writer, $util = $protobuf.util;
    
    // Exported root namespace
    var $root = $protobuf.roots.tf || ($protobuf.roots.tf = {});
    
    $root.tensorflow = (function() {
    
        /**
         * Namespace tensorflow.
         * @exports tensorflow
         * @namespace
         */
        var tensorflow = {};
    
        tensorflow.SavedModel = (function() {
    
            /**
             * Properties of a SavedModel.
             * @memberof tensorflow
             * @interface ISavedModel
             * @property {number|Long|null} [savedModelSchemaVersion] SavedModel savedModelSchemaVersion
             * @property {Array.<tensorflow.IMetaGraphDef>|null} [metaGraphs] SavedModel metaGraphs
             */
    
            /**
             * Constructs a new SavedModel.
             * @memberof tensorflow
             * @classdesc Represents a SavedModel.
             * @implements ISavedModel
             * @constructor
             * @param {tensorflow.ISavedModel=} [properties] Properties to set
             */
            function SavedModel(properties) {
                this.metaGraphs = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * SavedModel savedModelSchemaVersion.
             * @member {number|Long} savedModelSchemaVersion
             * @memberof tensorflow.SavedModel
             * @instance
             */
            SavedModel.prototype.savedModelSchemaVersion = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
            /**
             * SavedModel metaGraphs.
             * @member {Array.<tensorflow.IMetaGraphDef>} metaGraphs
             * @memberof tensorflow.SavedModel
             * @instance
             */
            SavedModel.prototype.metaGraphs = $util.emptyArray;
    
            /**
             * Creates a new SavedModel instance using the specified properties.
             * @function create
             * @memberof tensorflow.SavedModel
             * @static
             * @param {tensorflow.ISavedModel=} [properties] Properties to set
             * @returns {tensorflow.SavedModel} SavedModel instance
             */
            SavedModel.create = function create(properties) {
                return new SavedModel(properties);
            };
    
            /**
             * Encodes the specified SavedModel message. Does not implicitly {@link tensorflow.SavedModel.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.SavedModel
             * @static
             * @param {tensorflow.ISavedModel} message SavedModel message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            SavedModel.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.savedModelSchemaVersion != null && message.hasOwnProperty("savedModelSchemaVersion"))
                    writer.uint32(/* id 1, wireType 0 =*/8).int64(message.savedModelSchemaVersion);
                if (message.metaGraphs != null && message.metaGraphs.length)
                    for (var i = 0; i < message.metaGraphs.length; ++i)
                        $root.tensorflow.MetaGraphDef.encode(message.metaGraphs[i], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                return writer;
            };
    
            /**
             * Encodes the specified SavedModel message, length delimited. Does not implicitly {@link tensorflow.SavedModel.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.SavedModel
             * @static
             * @param {tensorflow.ISavedModel} message SavedModel message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            SavedModel.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a SavedModel message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.SavedModel
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.SavedModel} SavedModel
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a SavedModel message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.SavedModel
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.SavedModel} SavedModel
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            SavedModel.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a SavedModel message.
             * @function verify
             * @memberof tensorflow.SavedModel
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a SavedModel message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.SavedModel
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.SavedModel} SavedModel
             */
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
    
            /**
             * Creates a plain object from a SavedModel message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.SavedModel
             * @static
             * @param {tensorflow.SavedModel} message SavedModel
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this SavedModel to JSON.
             * @function toJSON
             * @memberof tensorflow.SavedModel
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            SavedModel.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return SavedModel;
        })();
    
        tensorflow.MetaGraphDef = (function() {
    
            /**
             * Properties of a MetaGraphDef.
             * @memberof tensorflow
             * @interface IMetaGraphDef
             * @property {tensorflow.MetaGraphDef.IMetaInfoDef|null} [metaInfoDef] MetaGraphDef metaInfoDef
             * @property {tensorflow.IGraphDef|null} [graphDef] MetaGraphDef graphDef
             * @property {tensorflow.ISaverDef|null} [saverDef] MetaGraphDef saverDef
             * @property {Object.<string,tensorflow.ICollectionDef>|null} [collectionDef] MetaGraphDef collectionDef
             * @property {Object.<string,tensorflow.ISignatureDef>|null} [signatureDef] MetaGraphDef signatureDef
             * @property {Array.<tensorflow.IAssetFileDef>|null} [assetFileDef] MetaGraphDef assetFileDef
             */
    
            /**
             * Constructs a new MetaGraphDef.
             * @memberof tensorflow
             * @classdesc Represents a MetaGraphDef.
             * @implements IMetaGraphDef
             * @constructor
             * @param {tensorflow.IMetaGraphDef=} [properties] Properties to set
             */
            function MetaGraphDef(properties) {
                this.collectionDef = {};
                this.signatureDef = {};
                this.assetFileDef = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * MetaGraphDef metaInfoDef.
             * @member {tensorflow.MetaGraphDef.IMetaInfoDef|null|undefined} metaInfoDef
             * @memberof tensorflow.MetaGraphDef
             * @instance
             */
            MetaGraphDef.prototype.metaInfoDef = null;
    
            /**
             * MetaGraphDef graphDef.
             * @member {tensorflow.IGraphDef|null|undefined} graphDef
             * @memberof tensorflow.MetaGraphDef
             * @instance
             */
            MetaGraphDef.prototype.graphDef = null;
    
            /**
             * MetaGraphDef saverDef.
             * @member {tensorflow.ISaverDef|null|undefined} saverDef
             * @memberof tensorflow.MetaGraphDef
             * @instance
             */
            MetaGraphDef.prototype.saverDef = null;
    
            /**
             * MetaGraphDef collectionDef.
             * @member {Object.<string,tensorflow.ICollectionDef>} collectionDef
             * @memberof tensorflow.MetaGraphDef
             * @instance
             */
            MetaGraphDef.prototype.collectionDef = $util.emptyObject;
    
            /**
             * MetaGraphDef signatureDef.
             * @member {Object.<string,tensorflow.ISignatureDef>} signatureDef
             * @memberof tensorflow.MetaGraphDef
             * @instance
             */
            MetaGraphDef.prototype.signatureDef = $util.emptyObject;
    
            /**
             * MetaGraphDef assetFileDef.
             * @member {Array.<tensorflow.IAssetFileDef>} assetFileDef
             * @memberof tensorflow.MetaGraphDef
             * @instance
             */
            MetaGraphDef.prototype.assetFileDef = $util.emptyArray;
    
            /**
             * Creates a new MetaGraphDef instance using the specified properties.
             * @function create
             * @memberof tensorflow.MetaGraphDef
             * @static
             * @param {tensorflow.IMetaGraphDef=} [properties] Properties to set
             * @returns {tensorflow.MetaGraphDef} MetaGraphDef instance
             */
            MetaGraphDef.create = function create(properties) {
                return new MetaGraphDef(properties);
            };
    
            /**
             * Encodes the specified MetaGraphDef message. Does not implicitly {@link tensorflow.MetaGraphDef.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.MetaGraphDef
             * @static
             * @param {tensorflow.IMetaGraphDef} message MetaGraphDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            MetaGraphDef.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.metaInfoDef != null && message.hasOwnProperty("metaInfoDef"))
                    $root.tensorflow.MetaGraphDef.MetaInfoDef.encode(message.metaInfoDef, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                if (message.graphDef != null && message.hasOwnProperty("graphDef"))
                    $root.tensorflow.GraphDef.encode(message.graphDef, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                if (message.saverDef != null && message.hasOwnProperty("saverDef"))
                    $root.tensorflow.SaverDef.encode(message.saverDef, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
                if (message.collectionDef != null && message.hasOwnProperty("collectionDef"))
                    for (var keys = Object.keys(message.collectionDef), i = 0; i < keys.length; ++i) {
                        writer.uint32(/* id 4, wireType 2 =*/34).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]);
                        $root.tensorflow.CollectionDef.encode(message.collectionDef[keys[i]], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim().ldelim();
                    }
                if (message.signatureDef != null && message.hasOwnProperty("signatureDef"))
                    for (var keys = Object.keys(message.signatureDef), i = 0; i < keys.length; ++i) {
                        writer.uint32(/* id 5, wireType 2 =*/42).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]);
                        $root.tensorflow.SignatureDef.encode(message.signatureDef[keys[i]], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim().ldelim();
                    }
                if (message.assetFileDef != null && message.assetFileDef.length)
                    for (var i = 0; i < message.assetFileDef.length; ++i)
                        $root.tensorflow.AssetFileDef.encode(message.assetFileDef[i], writer.uint32(/* id 6, wireType 2 =*/50).fork()).ldelim();
                return writer;
            };
    
            /**
             * Encodes the specified MetaGraphDef message, length delimited. Does not implicitly {@link tensorflow.MetaGraphDef.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.MetaGraphDef
             * @static
             * @param {tensorflow.IMetaGraphDef} message MetaGraphDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            MetaGraphDef.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a MetaGraphDef message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.MetaGraphDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.MetaGraphDef} MetaGraphDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a MetaGraphDef message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.MetaGraphDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.MetaGraphDef} MetaGraphDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            MetaGraphDef.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a MetaGraphDef message.
             * @function verify
             * @memberof tensorflow.MetaGraphDef
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a MetaGraphDef message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.MetaGraphDef
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.MetaGraphDef} MetaGraphDef
             */
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
    
            /**
             * Creates a plain object from a MetaGraphDef message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.MetaGraphDef
             * @static
             * @param {tensorflow.MetaGraphDef} message MetaGraphDef
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this MetaGraphDef to JSON.
             * @function toJSON
             * @memberof tensorflow.MetaGraphDef
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            MetaGraphDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            MetaGraphDef.MetaInfoDef = (function() {
    
                /**
                 * Properties of a MetaInfoDef.
                 * @memberof tensorflow.MetaGraphDef
                 * @interface IMetaInfoDef
                 * @property {string|null} [metaGraphVersion] MetaInfoDef metaGraphVersion
                 * @property {tensorflow.IOpList|null} [strippedOpList] MetaInfoDef strippedOpList
                 * @property {google.protobuf.IAny|null} [anyInfo] MetaInfoDef anyInfo
                 * @property {Array.<string>|null} [tags] MetaInfoDef tags
                 * @property {string|null} [tensorflowVersion] MetaInfoDef tensorflowVersion
                 * @property {string|null} [tensorflowGitVersion] MetaInfoDef tensorflowGitVersion
                 * @property {boolean|null} [strippedDefaultAttrs] MetaInfoDef strippedDefaultAttrs
                 */
    
                /**
                 * Constructs a new MetaInfoDef.
                 * @memberof tensorflow.MetaGraphDef
                 * @classdesc Represents a MetaInfoDef.
                 * @implements IMetaInfoDef
                 * @constructor
                 * @param {tensorflow.MetaGraphDef.IMetaInfoDef=} [properties] Properties to set
                 */
                function MetaInfoDef(properties) {
                    this.tags = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * MetaInfoDef metaGraphVersion.
                 * @member {string} metaGraphVersion
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @instance
                 */
                MetaInfoDef.prototype.metaGraphVersion = "";
    
                /**
                 * MetaInfoDef strippedOpList.
                 * @member {tensorflow.IOpList|null|undefined} strippedOpList
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @instance
                 */
                MetaInfoDef.prototype.strippedOpList = null;
    
                /**
                 * MetaInfoDef anyInfo.
                 * @member {google.protobuf.IAny|null|undefined} anyInfo
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @instance
                 */
                MetaInfoDef.prototype.anyInfo = null;
    
                /**
                 * MetaInfoDef tags.
                 * @member {Array.<string>} tags
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @instance
                 */
                MetaInfoDef.prototype.tags = $util.emptyArray;
    
                /**
                 * MetaInfoDef tensorflowVersion.
                 * @member {string} tensorflowVersion
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @instance
                 */
                MetaInfoDef.prototype.tensorflowVersion = "";
    
                /**
                 * MetaInfoDef tensorflowGitVersion.
                 * @member {string} tensorflowGitVersion
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @instance
                 */
                MetaInfoDef.prototype.tensorflowGitVersion = "";
    
                /**
                 * MetaInfoDef strippedDefaultAttrs.
                 * @member {boolean} strippedDefaultAttrs
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @instance
                 */
                MetaInfoDef.prototype.strippedDefaultAttrs = false;
    
                /**
                 * Creates a new MetaInfoDef instance using the specified properties.
                 * @function create
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @static
                 * @param {tensorflow.MetaGraphDef.IMetaInfoDef=} [properties] Properties to set
                 * @returns {tensorflow.MetaGraphDef.MetaInfoDef} MetaInfoDef instance
                 */
                MetaInfoDef.create = function create(properties) {
                    return new MetaInfoDef(properties);
                };
    
                /**
                 * Encodes the specified MetaInfoDef message. Does not implicitly {@link tensorflow.MetaGraphDef.MetaInfoDef.verify|verify} messages.
                 * @function encode
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @static
                 * @param {tensorflow.MetaGraphDef.IMetaInfoDef} message MetaInfoDef message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                MetaInfoDef.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.metaGraphVersion != null && message.hasOwnProperty("metaGraphVersion"))
                        writer.uint32(/* id 1, wireType 2 =*/10).string(message.metaGraphVersion);
                    if (message.strippedOpList != null && message.hasOwnProperty("strippedOpList"))
                        $root.tensorflow.OpList.encode(message.strippedOpList, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                    if (message.anyInfo != null && message.hasOwnProperty("anyInfo"))
                        $root.google.protobuf.Any.encode(message.anyInfo, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
                    if (message.tags != null && message.tags.length)
                        for (var i = 0; i < message.tags.length; ++i)
                            writer.uint32(/* id 4, wireType 2 =*/34).string(message.tags[i]);
                    if (message.tensorflowVersion != null && message.hasOwnProperty("tensorflowVersion"))
                        writer.uint32(/* id 5, wireType 2 =*/42).string(message.tensorflowVersion);
                    if (message.tensorflowGitVersion != null && message.hasOwnProperty("tensorflowGitVersion"))
                        writer.uint32(/* id 6, wireType 2 =*/50).string(message.tensorflowGitVersion);
                    if (message.strippedDefaultAttrs != null && message.hasOwnProperty("strippedDefaultAttrs"))
                        writer.uint32(/* id 7, wireType 0 =*/56).bool(message.strippedDefaultAttrs);
                    return writer;
                };
    
                /**
                 * Encodes the specified MetaInfoDef message, length delimited. Does not implicitly {@link tensorflow.MetaGraphDef.MetaInfoDef.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @static
                 * @param {tensorflow.MetaGraphDef.IMetaInfoDef} message MetaInfoDef message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                MetaInfoDef.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes a MetaInfoDef message from the specified reader or buffer.
                 * @function decode
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {tensorflow.MetaGraphDef.MetaInfoDef} MetaInfoDef
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
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
    
                /**
                 * Decodes a MetaInfoDef message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {tensorflow.MetaGraphDef.MetaInfoDef} MetaInfoDef
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                MetaInfoDef.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies a MetaInfoDef message.
                 * @function verify
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
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
    
                /**
                 * Creates a MetaInfoDef message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {tensorflow.MetaGraphDef.MetaInfoDef} MetaInfoDef
                 */
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
    
                /**
                 * Creates a plain object from a MetaInfoDef message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @static
                 * @param {tensorflow.MetaGraphDef.MetaInfoDef} message MetaInfoDef
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
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
    
                /**
                 * Converts this MetaInfoDef to JSON.
                 * @function toJSON
                 * @memberof tensorflow.MetaGraphDef.MetaInfoDef
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                MetaInfoDef.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return MetaInfoDef;
            })();
    
            return MetaGraphDef;
        })();
    
        tensorflow.CollectionDef = (function() {
    
            /**
             * Properties of a CollectionDef.
             * @memberof tensorflow
             * @interface ICollectionDef
             * @property {tensorflow.CollectionDef.INodeList|null} [nodeList] CollectionDef nodeList
             * @property {tensorflow.CollectionDef.IBytesList|null} [bytesList] CollectionDef bytesList
             * @property {tensorflow.CollectionDef.IInt64List|null} [int64List] CollectionDef int64List
             * @property {tensorflow.CollectionDef.IFloatList|null} [floatList] CollectionDef floatList
             * @property {tensorflow.CollectionDef.IAnyList|null} [anyList] CollectionDef anyList
             */
    
            /**
             * Constructs a new CollectionDef.
             * @memberof tensorflow
             * @classdesc Represents a CollectionDef.
             * @implements ICollectionDef
             * @constructor
             * @param {tensorflow.ICollectionDef=} [properties] Properties to set
             */
            function CollectionDef(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * CollectionDef nodeList.
             * @member {tensorflow.CollectionDef.INodeList|null|undefined} nodeList
             * @memberof tensorflow.CollectionDef
             * @instance
             */
            CollectionDef.prototype.nodeList = null;
    
            /**
             * CollectionDef bytesList.
             * @member {tensorflow.CollectionDef.IBytesList|null|undefined} bytesList
             * @memberof tensorflow.CollectionDef
             * @instance
             */
            CollectionDef.prototype.bytesList = null;
    
            /**
             * CollectionDef int64List.
             * @member {tensorflow.CollectionDef.IInt64List|null|undefined} int64List
             * @memberof tensorflow.CollectionDef
             * @instance
             */
            CollectionDef.prototype.int64List = null;
    
            /**
             * CollectionDef floatList.
             * @member {tensorflow.CollectionDef.IFloatList|null|undefined} floatList
             * @memberof tensorflow.CollectionDef
             * @instance
             */
            CollectionDef.prototype.floatList = null;
    
            /**
             * CollectionDef anyList.
             * @member {tensorflow.CollectionDef.IAnyList|null|undefined} anyList
             * @memberof tensorflow.CollectionDef
             * @instance
             */
            CollectionDef.prototype.anyList = null;
    
            // OneOf field names bound to virtual getters and setters
            var $oneOfFields;
    
            /**
             * CollectionDef kind.
             * @member {"nodeList"|"bytesList"|"int64List"|"floatList"|"anyList"|undefined} kind
             * @memberof tensorflow.CollectionDef
             * @instance
             */
            Object.defineProperty(CollectionDef.prototype, "kind", {
                get: $util.oneOfGetter($oneOfFields = ["nodeList", "bytesList", "int64List", "floatList", "anyList"]),
                set: $util.oneOfSetter($oneOfFields)
            });
    
            /**
             * Creates a new CollectionDef instance using the specified properties.
             * @function create
             * @memberof tensorflow.CollectionDef
             * @static
             * @param {tensorflow.ICollectionDef=} [properties] Properties to set
             * @returns {tensorflow.CollectionDef} CollectionDef instance
             */
            CollectionDef.create = function create(properties) {
                return new CollectionDef(properties);
            };
    
            /**
             * Encodes the specified CollectionDef message. Does not implicitly {@link tensorflow.CollectionDef.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.CollectionDef
             * @static
             * @param {tensorflow.ICollectionDef} message CollectionDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            CollectionDef.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.nodeList != null && message.hasOwnProperty("nodeList"))
                    $root.tensorflow.CollectionDef.NodeList.encode(message.nodeList, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                if (message.bytesList != null && message.hasOwnProperty("bytesList"))
                    $root.tensorflow.CollectionDef.BytesList.encode(message.bytesList, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                if (message.int64List != null && message.hasOwnProperty("int64List"))
                    $root.tensorflow.CollectionDef.Int64List.encode(message.int64List, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
                if (message.floatList != null && message.hasOwnProperty("floatList"))
                    $root.tensorflow.CollectionDef.FloatList.encode(message.floatList, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
                if (message.anyList != null && message.hasOwnProperty("anyList"))
                    $root.tensorflow.CollectionDef.AnyList.encode(message.anyList, writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
                return writer;
            };
    
            /**
             * Encodes the specified CollectionDef message, length delimited. Does not implicitly {@link tensorflow.CollectionDef.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.CollectionDef
             * @static
             * @param {tensorflow.ICollectionDef} message CollectionDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            CollectionDef.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a CollectionDef message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.CollectionDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.CollectionDef} CollectionDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a CollectionDef message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.CollectionDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.CollectionDef} CollectionDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            CollectionDef.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a CollectionDef message.
             * @function verify
             * @memberof tensorflow.CollectionDef
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a CollectionDef message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.CollectionDef
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.CollectionDef} CollectionDef
             */
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
    
            /**
             * Creates a plain object from a CollectionDef message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.CollectionDef
             * @static
             * @param {tensorflow.CollectionDef} message CollectionDef
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this CollectionDef to JSON.
             * @function toJSON
             * @memberof tensorflow.CollectionDef
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            CollectionDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            CollectionDef.NodeList = (function() {
    
                /**
                 * Properties of a NodeList.
                 * @memberof tensorflow.CollectionDef
                 * @interface INodeList
                 * @property {Array.<string>|null} [value] NodeList value
                 */
    
                /**
                 * Constructs a new NodeList.
                 * @memberof tensorflow.CollectionDef
                 * @classdesc Represents a NodeList.
                 * @implements INodeList
                 * @constructor
                 * @param {tensorflow.CollectionDef.INodeList=} [properties] Properties to set
                 */
                function NodeList(properties) {
                    this.value = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * NodeList value.
                 * @member {Array.<string>} value
                 * @memberof tensorflow.CollectionDef.NodeList
                 * @instance
                 */
                NodeList.prototype.value = $util.emptyArray;
    
                /**
                 * Creates a new NodeList instance using the specified properties.
                 * @function create
                 * @memberof tensorflow.CollectionDef.NodeList
                 * @static
                 * @param {tensorflow.CollectionDef.INodeList=} [properties] Properties to set
                 * @returns {tensorflow.CollectionDef.NodeList} NodeList instance
                 */
                NodeList.create = function create(properties) {
                    return new NodeList(properties);
                };
    
                /**
                 * Encodes the specified NodeList message. Does not implicitly {@link tensorflow.CollectionDef.NodeList.verify|verify} messages.
                 * @function encode
                 * @memberof tensorflow.CollectionDef.NodeList
                 * @static
                 * @param {tensorflow.CollectionDef.INodeList} message NodeList message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                NodeList.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.value != null && message.value.length)
                        for (var i = 0; i < message.value.length; ++i)
                            writer.uint32(/* id 1, wireType 2 =*/10).string(message.value[i]);
                    return writer;
                };
    
                /**
                 * Encodes the specified NodeList message, length delimited. Does not implicitly {@link tensorflow.CollectionDef.NodeList.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof tensorflow.CollectionDef.NodeList
                 * @static
                 * @param {tensorflow.CollectionDef.INodeList} message NodeList message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                NodeList.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes a NodeList message from the specified reader or buffer.
                 * @function decode
                 * @memberof tensorflow.CollectionDef.NodeList
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {tensorflow.CollectionDef.NodeList} NodeList
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
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
    
                /**
                 * Decodes a NodeList message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof tensorflow.CollectionDef.NodeList
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {tensorflow.CollectionDef.NodeList} NodeList
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                NodeList.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies a NodeList message.
                 * @function verify
                 * @memberof tensorflow.CollectionDef.NodeList
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
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
    
                /**
                 * Creates a NodeList message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof tensorflow.CollectionDef.NodeList
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {tensorflow.CollectionDef.NodeList} NodeList
                 */
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
    
                /**
                 * Creates a plain object from a NodeList message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof tensorflow.CollectionDef.NodeList
                 * @static
                 * @param {tensorflow.CollectionDef.NodeList} message NodeList
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
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
    
                /**
                 * Converts this NodeList to JSON.
                 * @function toJSON
                 * @memberof tensorflow.CollectionDef.NodeList
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                NodeList.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return NodeList;
            })();
    
            CollectionDef.BytesList = (function() {
    
                /**
                 * Properties of a BytesList.
                 * @memberof tensorflow.CollectionDef
                 * @interface IBytesList
                 * @property {Array.<Uint8Array>|null} [value] BytesList value
                 */
    
                /**
                 * Constructs a new BytesList.
                 * @memberof tensorflow.CollectionDef
                 * @classdesc Represents a BytesList.
                 * @implements IBytesList
                 * @constructor
                 * @param {tensorflow.CollectionDef.IBytesList=} [properties] Properties to set
                 */
                function BytesList(properties) {
                    this.value = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * BytesList value.
                 * @member {Array.<Uint8Array>} value
                 * @memberof tensorflow.CollectionDef.BytesList
                 * @instance
                 */
                BytesList.prototype.value = $util.emptyArray;
    
                /**
                 * Creates a new BytesList instance using the specified properties.
                 * @function create
                 * @memberof tensorflow.CollectionDef.BytesList
                 * @static
                 * @param {tensorflow.CollectionDef.IBytesList=} [properties] Properties to set
                 * @returns {tensorflow.CollectionDef.BytesList} BytesList instance
                 */
                BytesList.create = function create(properties) {
                    return new BytesList(properties);
                };
    
                /**
                 * Encodes the specified BytesList message. Does not implicitly {@link tensorflow.CollectionDef.BytesList.verify|verify} messages.
                 * @function encode
                 * @memberof tensorflow.CollectionDef.BytesList
                 * @static
                 * @param {tensorflow.CollectionDef.IBytesList} message BytesList message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                BytesList.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.value != null && message.value.length)
                        for (var i = 0; i < message.value.length; ++i)
                            writer.uint32(/* id 1, wireType 2 =*/10).bytes(message.value[i]);
                    return writer;
                };
    
                /**
                 * Encodes the specified BytesList message, length delimited. Does not implicitly {@link tensorflow.CollectionDef.BytesList.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof tensorflow.CollectionDef.BytesList
                 * @static
                 * @param {tensorflow.CollectionDef.IBytesList} message BytesList message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                BytesList.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes a BytesList message from the specified reader or buffer.
                 * @function decode
                 * @memberof tensorflow.CollectionDef.BytesList
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {tensorflow.CollectionDef.BytesList} BytesList
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
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
    
                /**
                 * Decodes a BytesList message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof tensorflow.CollectionDef.BytesList
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {tensorflow.CollectionDef.BytesList} BytesList
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                BytesList.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies a BytesList message.
                 * @function verify
                 * @memberof tensorflow.CollectionDef.BytesList
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
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
    
                /**
                 * Creates a BytesList message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof tensorflow.CollectionDef.BytesList
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {tensorflow.CollectionDef.BytesList} BytesList
                 */
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
    
                /**
                 * Creates a plain object from a BytesList message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof tensorflow.CollectionDef.BytesList
                 * @static
                 * @param {tensorflow.CollectionDef.BytesList} message BytesList
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
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
    
                /**
                 * Converts this BytesList to JSON.
                 * @function toJSON
                 * @memberof tensorflow.CollectionDef.BytesList
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                BytesList.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return BytesList;
            })();
    
            CollectionDef.Int64List = (function() {
    
                /**
                 * Properties of an Int64List.
                 * @memberof tensorflow.CollectionDef
                 * @interface IInt64List
                 * @property {Array.<number|Long>|null} [value] Int64List value
                 */
    
                /**
                 * Constructs a new Int64List.
                 * @memberof tensorflow.CollectionDef
                 * @classdesc Represents an Int64List.
                 * @implements IInt64List
                 * @constructor
                 * @param {tensorflow.CollectionDef.IInt64List=} [properties] Properties to set
                 */
                function Int64List(properties) {
                    this.value = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * Int64List value.
                 * @member {Array.<number|Long>} value
                 * @memberof tensorflow.CollectionDef.Int64List
                 * @instance
                 */
                Int64List.prototype.value = $util.emptyArray;
    
                /**
                 * Creates a new Int64List instance using the specified properties.
                 * @function create
                 * @memberof tensorflow.CollectionDef.Int64List
                 * @static
                 * @param {tensorflow.CollectionDef.IInt64List=} [properties] Properties to set
                 * @returns {tensorflow.CollectionDef.Int64List} Int64List instance
                 */
                Int64List.create = function create(properties) {
                    return new Int64List(properties);
                };
    
                /**
                 * Encodes the specified Int64List message. Does not implicitly {@link tensorflow.CollectionDef.Int64List.verify|verify} messages.
                 * @function encode
                 * @memberof tensorflow.CollectionDef.Int64List
                 * @static
                 * @param {tensorflow.CollectionDef.IInt64List} message Int64List message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Int64List.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.value != null && message.value.length) {
                        writer.uint32(/* id 1, wireType 2 =*/10).fork();
                        for (var i = 0; i < message.value.length; ++i)
                            writer.int64(message.value[i]);
                        writer.ldelim();
                    }
                    return writer;
                };
    
                /**
                 * Encodes the specified Int64List message, length delimited. Does not implicitly {@link tensorflow.CollectionDef.Int64List.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof tensorflow.CollectionDef.Int64List
                 * @static
                 * @param {tensorflow.CollectionDef.IInt64List} message Int64List message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Int64List.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes an Int64List message from the specified reader or buffer.
                 * @function decode
                 * @memberof tensorflow.CollectionDef.Int64List
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {tensorflow.CollectionDef.Int64List} Int64List
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
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
    
                /**
                 * Decodes an Int64List message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof tensorflow.CollectionDef.Int64List
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {tensorflow.CollectionDef.Int64List} Int64List
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                Int64List.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies an Int64List message.
                 * @function verify
                 * @memberof tensorflow.CollectionDef.Int64List
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
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
    
                /**
                 * Creates an Int64List message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof tensorflow.CollectionDef.Int64List
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {tensorflow.CollectionDef.Int64List} Int64List
                 */
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
    
                /**
                 * Creates a plain object from an Int64List message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof tensorflow.CollectionDef.Int64List
                 * @static
                 * @param {tensorflow.CollectionDef.Int64List} message Int64List
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
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
    
                /**
                 * Converts this Int64List to JSON.
                 * @function toJSON
                 * @memberof tensorflow.CollectionDef.Int64List
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                Int64List.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return Int64List;
            })();
    
            CollectionDef.FloatList = (function() {
    
                /**
                 * Properties of a FloatList.
                 * @memberof tensorflow.CollectionDef
                 * @interface IFloatList
                 * @property {Array.<number>|null} [value] FloatList value
                 */
    
                /**
                 * Constructs a new FloatList.
                 * @memberof tensorflow.CollectionDef
                 * @classdesc Represents a FloatList.
                 * @implements IFloatList
                 * @constructor
                 * @param {tensorflow.CollectionDef.IFloatList=} [properties] Properties to set
                 */
                function FloatList(properties) {
                    this.value = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * FloatList value.
                 * @member {Array.<number>} value
                 * @memberof tensorflow.CollectionDef.FloatList
                 * @instance
                 */
                FloatList.prototype.value = $util.emptyArray;
    
                /**
                 * Creates a new FloatList instance using the specified properties.
                 * @function create
                 * @memberof tensorflow.CollectionDef.FloatList
                 * @static
                 * @param {tensorflow.CollectionDef.IFloatList=} [properties] Properties to set
                 * @returns {tensorflow.CollectionDef.FloatList} FloatList instance
                 */
                FloatList.create = function create(properties) {
                    return new FloatList(properties);
                };
    
                /**
                 * Encodes the specified FloatList message. Does not implicitly {@link tensorflow.CollectionDef.FloatList.verify|verify} messages.
                 * @function encode
                 * @memberof tensorflow.CollectionDef.FloatList
                 * @static
                 * @param {tensorflow.CollectionDef.IFloatList} message FloatList message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                FloatList.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.value != null && message.value.length) {
                        writer.uint32(/* id 1, wireType 2 =*/10).fork();
                        for (var i = 0; i < message.value.length; ++i)
                            writer.float(message.value[i]);
                        writer.ldelim();
                    }
                    return writer;
                };
    
                /**
                 * Encodes the specified FloatList message, length delimited. Does not implicitly {@link tensorflow.CollectionDef.FloatList.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof tensorflow.CollectionDef.FloatList
                 * @static
                 * @param {tensorflow.CollectionDef.IFloatList} message FloatList message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                FloatList.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes a FloatList message from the specified reader or buffer.
                 * @function decode
                 * @memberof tensorflow.CollectionDef.FloatList
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {tensorflow.CollectionDef.FloatList} FloatList
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
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
    
                /**
                 * Decodes a FloatList message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof tensorflow.CollectionDef.FloatList
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {tensorflow.CollectionDef.FloatList} FloatList
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                FloatList.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies a FloatList message.
                 * @function verify
                 * @memberof tensorflow.CollectionDef.FloatList
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
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
    
                /**
                 * Creates a FloatList message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof tensorflow.CollectionDef.FloatList
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {tensorflow.CollectionDef.FloatList} FloatList
                 */
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
    
                /**
                 * Creates a plain object from a FloatList message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof tensorflow.CollectionDef.FloatList
                 * @static
                 * @param {tensorflow.CollectionDef.FloatList} message FloatList
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
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
    
                /**
                 * Converts this FloatList to JSON.
                 * @function toJSON
                 * @memberof tensorflow.CollectionDef.FloatList
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                FloatList.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return FloatList;
            })();
    
            CollectionDef.AnyList = (function() {
    
                /**
                 * Properties of an AnyList.
                 * @memberof tensorflow.CollectionDef
                 * @interface IAnyList
                 * @property {Array.<google.protobuf.IAny>|null} [value] AnyList value
                 */
    
                /**
                 * Constructs a new AnyList.
                 * @memberof tensorflow.CollectionDef
                 * @classdesc Represents an AnyList.
                 * @implements IAnyList
                 * @constructor
                 * @param {tensorflow.CollectionDef.IAnyList=} [properties] Properties to set
                 */
                function AnyList(properties) {
                    this.value = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * AnyList value.
                 * @member {Array.<google.protobuf.IAny>} value
                 * @memberof tensorflow.CollectionDef.AnyList
                 * @instance
                 */
                AnyList.prototype.value = $util.emptyArray;
    
                /**
                 * Creates a new AnyList instance using the specified properties.
                 * @function create
                 * @memberof tensorflow.CollectionDef.AnyList
                 * @static
                 * @param {tensorflow.CollectionDef.IAnyList=} [properties] Properties to set
                 * @returns {tensorflow.CollectionDef.AnyList} AnyList instance
                 */
                AnyList.create = function create(properties) {
                    return new AnyList(properties);
                };
    
                /**
                 * Encodes the specified AnyList message. Does not implicitly {@link tensorflow.CollectionDef.AnyList.verify|verify} messages.
                 * @function encode
                 * @memberof tensorflow.CollectionDef.AnyList
                 * @static
                 * @param {tensorflow.CollectionDef.IAnyList} message AnyList message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                AnyList.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.value != null && message.value.length)
                        for (var i = 0; i < message.value.length; ++i)
                            $root.google.protobuf.Any.encode(message.value[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                    return writer;
                };
    
                /**
                 * Encodes the specified AnyList message, length delimited. Does not implicitly {@link tensorflow.CollectionDef.AnyList.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof tensorflow.CollectionDef.AnyList
                 * @static
                 * @param {tensorflow.CollectionDef.IAnyList} message AnyList message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                AnyList.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes an AnyList message from the specified reader or buffer.
                 * @function decode
                 * @memberof tensorflow.CollectionDef.AnyList
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {tensorflow.CollectionDef.AnyList} AnyList
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
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
    
                /**
                 * Decodes an AnyList message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof tensorflow.CollectionDef.AnyList
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {tensorflow.CollectionDef.AnyList} AnyList
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                AnyList.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies an AnyList message.
                 * @function verify
                 * @memberof tensorflow.CollectionDef.AnyList
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
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
    
                /**
                 * Creates an AnyList message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof tensorflow.CollectionDef.AnyList
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {tensorflow.CollectionDef.AnyList} AnyList
                 */
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
    
                /**
                 * Creates a plain object from an AnyList message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof tensorflow.CollectionDef.AnyList
                 * @static
                 * @param {tensorflow.CollectionDef.AnyList} message AnyList
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
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
    
                /**
                 * Converts this AnyList to JSON.
                 * @function toJSON
                 * @memberof tensorflow.CollectionDef.AnyList
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                AnyList.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return AnyList;
            })();
    
            return CollectionDef;
        })();
    
        tensorflow.TensorInfo = (function() {
    
            /**
             * Properties of a TensorInfo.
             * @memberof tensorflow
             * @interface ITensorInfo
             * @property {string|null} [name] TensorInfo name
             * @property {tensorflow.TensorInfo.ICooSparse|null} [cooSparse] TensorInfo cooSparse
             * @property {tensorflow.DataType|null} [dtype] TensorInfo dtype
             * @property {tensorflow.ITensorShapeProto|null} [tensorShape] TensorInfo tensorShape
             */
    
            /**
             * Constructs a new TensorInfo.
             * @memberof tensorflow
             * @classdesc Represents a TensorInfo.
             * @implements ITensorInfo
             * @constructor
             * @param {tensorflow.ITensorInfo=} [properties] Properties to set
             */
            function TensorInfo(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * TensorInfo name.
             * @member {string} name
             * @memberof tensorflow.TensorInfo
             * @instance
             */
            TensorInfo.prototype.name = "";
    
            /**
             * TensorInfo cooSparse.
             * @member {tensorflow.TensorInfo.ICooSparse|null|undefined} cooSparse
             * @memberof tensorflow.TensorInfo
             * @instance
             */
            TensorInfo.prototype.cooSparse = null;
    
            /**
             * TensorInfo dtype.
             * @member {tensorflow.DataType} dtype
             * @memberof tensorflow.TensorInfo
             * @instance
             */
            TensorInfo.prototype.dtype = 0;
    
            /**
             * TensorInfo tensorShape.
             * @member {tensorflow.ITensorShapeProto|null|undefined} tensorShape
             * @memberof tensorflow.TensorInfo
             * @instance
             */
            TensorInfo.prototype.tensorShape = null;
    
            // OneOf field names bound to virtual getters and setters
            var $oneOfFields;
    
            /**
             * TensorInfo encoding.
             * @member {"name"|"cooSparse"|undefined} encoding
             * @memberof tensorflow.TensorInfo
             * @instance
             */
            Object.defineProperty(TensorInfo.prototype, "encoding", {
                get: $util.oneOfGetter($oneOfFields = ["name", "cooSparse"]),
                set: $util.oneOfSetter($oneOfFields)
            });
    
            /**
             * Creates a new TensorInfo instance using the specified properties.
             * @function create
             * @memberof tensorflow.TensorInfo
             * @static
             * @param {tensorflow.ITensorInfo=} [properties] Properties to set
             * @returns {tensorflow.TensorInfo} TensorInfo instance
             */
            TensorInfo.create = function create(properties) {
                return new TensorInfo(properties);
            };
    
            /**
             * Encodes the specified TensorInfo message. Does not implicitly {@link tensorflow.TensorInfo.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.TensorInfo
             * @static
             * @param {tensorflow.ITensorInfo} message TensorInfo message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TensorInfo.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.name != null && message.hasOwnProperty("name"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
                if (message.dtype != null && message.hasOwnProperty("dtype"))
                    writer.uint32(/* id 2, wireType 0 =*/16).int32(message.dtype);
                if (message.tensorShape != null && message.hasOwnProperty("tensorShape"))
                    $root.tensorflow.TensorShapeProto.encode(message.tensorShape, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
                if (message.cooSparse != null && message.hasOwnProperty("cooSparse"))
                    $root.tensorflow.TensorInfo.CooSparse.encode(message.cooSparse, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
                return writer;
            };
    
            /**
             * Encodes the specified TensorInfo message, length delimited. Does not implicitly {@link tensorflow.TensorInfo.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.TensorInfo
             * @static
             * @param {tensorflow.ITensorInfo} message TensorInfo message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TensorInfo.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a TensorInfo message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.TensorInfo
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.TensorInfo} TensorInfo
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a TensorInfo message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.TensorInfo
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.TensorInfo} TensorInfo
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            TensorInfo.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a TensorInfo message.
             * @function verify
             * @memberof tensorflow.TensorInfo
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a TensorInfo message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.TensorInfo
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.TensorInfo} TensorInfo
             */
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
    
            /**
             * Creates a plain object from a TensorInfo message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.TensorInfo
             * @static
             * @param {tensorflow.TensorInfo} message TensorInfo
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this TensorInfo to JSON.
             * @function toJSON
             * @memberof tensorflow.TensorInfo
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            TensorInfo.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            TensorInfo.CooSparse = (function() {
    
                /**
                 * Properties of a CooSparse.
                 * @memberof tensorflow.TensorInfo
                 * @interface ICooSparse
                 * @property {string|null} [valuesTensorName] CooSparse valuesTensorName
                 * @property {string|null} [indicesTensorName] CooSparse indicesTensorName
                 * @property {string|null} [denseShapeTensorName] CooSparse denseShapeTensorName
                 */
    
                /**
                 * Constructs a new CooSparse.
                 * @memberof tensorflow.TensorInfo
                 * @classdesc Represents a CooSparse.
                 * @implements ICooSparse
                 * @constructor
                 * @param {tensorflow.TensorInfo.ICooSparse=} [properties] Properties to set
                 */
                function CooSparse(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * CooSparse valuesTensorName.
                 * @member {string} valuesTensorName
                 * @memberof tensorflow.TensorInfo.CooSparse
                 * @instance
                 */
                CooSparse.prototype.valuesTensorName = "";
    
                /**
                 * CooSparse indicesTensorName.
                 * @member {string} indicesTensorName
                 * @memberof tensorflow.TensorInfo.CooSparse
                 * @instance
                 */
                CooSparse.prototype.indicesTensorName = "";
    
                /**
                 * CooSparse denseShapeTensorName.
                 * @member {string} denseShapeTensorName
                 * @memberof tensorflow.TensorInfo.CooSparse
                 * @instance
                 */
                CooSparse.prototype.denseShapeTensorName = "";
    
                /**
                 * Creates a new CooSparse instance using the specified properties.
                 * @function create
                 * @memberof tensorflow.TensorInfo.CooSparse
                 * @static
                 * @param {tensorflow.TensorInfo.ICooSparse=} [properties] Properties to set
                 * @returns {tensorflow.TensorInfo.CooSparse} CooSparse instance
                 */
                CooSparse.create = function create(properties) {
                    return new CooSparse(properties);
                };
    
                /**
                 * Encodes the specified CooSparse message. Does not implicitly {@link tensorflow.TensorInfo.CooSparse.verify|verify} messages.
                 * @function encode
                 * @memberof tensorflow.TensorInfo.CooSparse
                 * @static
                 * @param {tensorflow.TensorInfo.ICooSparse} message CooSparse message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                CooSparse.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.valuesTensorName != null && message.hasOwnProperty("valuesTensorName"))
                        writer.uint32(/* id 1, wireType 2 =*/10).string(message.valuesTensorName);
                    if (message.indicesTensorName != null && message.hasOwnProperty("indicesTensorName"))
                        writer.uint32(/* id 2, wireType 2 =*/18).string(message.indicesTensorName);
                    if (message.denseShapeTensorName != null && message.hasOwnProperty("denseShapeTensorName"))
                        writer.uint32(/* id 3, wireType 2 =*/26).string(message.denseShapeTensorName);
                    return writer;
                };
    
                /**
                 * Encodes the specified CooSparse message, length delimited. Does not implicitly {@link tensorflow.TensorInfo.CooSparse.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof tensorflow.TensorInfo.CooSparse
                 * @static
                 * @param {tensorflow.TensorInfo.ICooSparse} message CooSparse message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                CooSparse.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes a CooSparse message from the specified reader or buffer.
                 * @function decode
                 * @memberof tensorflow.TensorInfo.CooSparse
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {tensorflow.TensorInfo.CooSparse} CooSparse
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
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
    
                /**
                 * Decodes a CooSparse message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof tensorflow.TensorInfo.CooSparse
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {tensorflow.TensorInfo.CooSparse} CooSparse
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                CooSparse.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies a CooSparse message.
                 * @function verify
                 * @memberof tensorflow.TensorInfo.CooSparse
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
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
    
                /**
                 * Creates a CooSparse message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof tensorflow.TensorInfo.CooSparse
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {tensorflow.TensorInfo.CooSparse} CooSparse
                 */
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
    
                /**
                 * Creates a plain object from a CooSparse message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof tensorflow.TensorInfo.CooSparse
                 * @static
                 * @param {tensorflow.TensorInfo.CooSparse} message CooSparse
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
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
    
                /**
                 * Converts this CooSparse to JSON.
                 * @function toJSON
                 * @memberof tensorflow.TensorInfo.CooSparse
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                CooSparse.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return CooSparse;
            })();
    
            return TensorInfo;
        })();
    
        tensorflow.SignatureDef = (function() {
    
            /**
             * Properties of a SignatureDef.
             * @memberof tensorflow
             * @interface ISignatureDef
             * @property {Object.<string,tensorflow.ITensorInfo>|null} [inputs] SignatureDef inputs
             * @property {Object.<string,tensorflow.ITensorInfo>|null} [outputs] SignatureDef outputs
             * @property {string|null} [methodName] SignatureDef methodName
             */
    
            /**
             * Constructs a new SignatureDef.
             * @memberof tensorflow
             * @classdesc Represents a SignatureDef.
             * @implements ISignatureDef
             * @constructor
             * @param {tensorflow.ISignatureDef=} [properties] Properties to set
             */
            function SignatureDef(properties) {
                this.inputs = {};
                this.outputs = {};
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * SignatureDef inputs.
             * @member {Object.<string,tensorflow.ITensorInfo>} inputs
             * @memberof tensorflow.SignatureDef
             * @instance
             */
            SignatureDef.prototype.inputs = $util.emptyObject;
    
            /**
             * SignatureDef outputs.
             * @member {Object.<string,tensorflow.ITensorInfo>} outputs
             * @memberof tensorflow.SignatureDef
             * @instance
             */
            SignatureDef.prototype.outputs = $util.emptyObject;
    
            /**
             * SignatureDef methodName.
             * @member {string} methodName
             * @memberof tensorflow.SignatureDef
             * @instance
             */
            SignatureDef.prototype.methodName = "";
    
            /**
             * Creates a new SignatureDef instance using the specified properties.
             * @function create
             * @memberof tensorflow.SignatureDef
             * @static
             * @param {tensorflow.ISignatureDef=} [properties] Properties to set
             * @returns {tensorflow.SignatureDef} SignatureDef instance
             */
            SignatureDef.create = function create(properties) {
                return new SignatureDef(properties);
            };
    
            /**
             * Encodes the specified SignatureDef message. Does not implicitly {@link tensorflow.SignatureDef.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.SignatureDef
             * @static
             * @param {tensorflow.ISignatureDef} message SignatureDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            SignatureDef.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.inputs != null && message.hasOwnProperty("inputs"))
                    for (var keys = Object.keys(message.inputs), i = 0; i < keys.length; ++i) {
                        writer.uint32(/* id 1, wireType 2 =*/10).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]);
                        $root.tensorflow.TensorInfo.encode(message.inputs[keys[i]], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim().ldelim();
                    }
                if (message.outputs != null && message.hasOwnProperty("outputs"))
                    for (var keys = Object.keys(message.outputs), i = 0; i < keys.length; ++i) {
                        writer.uint32(/* id 2, wireType 2 =*/18).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]);
                        $root.tensorflow.TensorInfo.encode(message.outputs[keys[i]], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim().ldelim();
                    }
                if (message.methodName != null && message.hasOwnProperty("methodName"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.methodName);
                return writer;
            };
    
            /**
             * Encodes the specified SignatureDef message, length delimited. Does not implicitly {@link tensorflow.SignatureDef.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.SignatureDef
             * @static
             * @param {tensorflow.ISignatureDef} message SignatureDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            SignatureDef.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a SignatureDef message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.SignatureDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.SignatureDef} SignatureDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a SignatureDef message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.SignatureDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.SignatureDef} SignatureDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            SignatureDef.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a SignatureDef message.
             * @function verify
             * @memberof tensorflow.SignatureDef
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a SignatureDef message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.SignatureDef
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.SignatureDef} SignatureDef
             */
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
    
            /**
             * Creates a plain object from a SignatureDef message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.SignatureDef
             * @static
             * @param {tensorflow.SignatureDef} message SignatureDef
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this SignatureDef to JSON.
             * @function toJSON
             * @memberof tensorflow.SignatureDef
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            SignatureDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return SignatureDef;
        })();
    
        tensorflow.AssetFileDef = (function() {
    
            /**
             * Properties of an AssetFileDef.
             * @memberof tensorflow
             * @interface IAssetFileDef
             * @property {tensorflow.ITensorInfo|null} [tensorInfo] AssetFileDef tensorInfo
             * @property {string|null} [filename] AssetFileDef filename
             */
    
            /**
             * Constructs a new AssetFileDef.
             * @memberof tensorflow
             * @classdesc Represents an AssetFileDef.
             * @implements IAssetFileDef
             * @constructor
             * @param {tensorflow.IAssetFileDef=} [properties] Properties to set
             */
            function AssetFileDef(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * AssetFileDef tensorInfo.
             * @member {tensorflow.ITensorInfo|null|undefined} tensorInfo
             * @memberof tensorflow.AssetFileDef
             * @instance
             */
            AssetFileDef.prototype.tensorInfo = null;
    
            /**
             * AssetFileDef filename.
             * @member {string} filename
             * @memberof tensorflow.AssetFileDef
             * @instance
             */
            AssetFileDef.prototype.filename = "";
    
            /**
             * Creates a new AssetFileDef instance using the specified properties.
             * @function create
             * @memberof tensorflow.AssetFileDef
             * @static
             * @param {tensorflow.IAssetFileDef=} [properties] Properties to set
             * @returns {tensorflow.AssetFileDef} AssetFileDef instance
             */
            AssetFileDef.create = function create(properties) {
                return new AssetFileDef(properties);
            };
    
            /**
             * Encodes the specified AssetFileDef message. Does not implicitly {@link tensorflow.AssetFileDef.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.AssetFileDef
             * @static
             * @param {tensorflow.IAssetFileDef} message AssetFileDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AssetFileDef.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.tensorInfo != null && message.hasOwnProperty("tensorInfo"))
                    $root.tensorflow.TensorInfo.encode(message.tensorInfo, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                if (message.filename != null && message.hasOwnProperty("filename"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.filename);
                return writer;
            };
    
            /**
             * Encodes the specified AssetFileDef message, length delimited. Does not implicitly {@link tensorflow.AssetFileDef.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.AssetFileDef
             * @static
             * @param {tensorflow.IAssetFileDef} message AssetFileDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AssetFileDef.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes an AssetFileDef message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.AssetFileDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.AssetFileDef} AssetFileDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes an AssetFileDef message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.AssetFileDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.AssetFileDef} AssetFileDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AssetFileDef.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies an AssetFileDef message.
             * @function verify
             * @memberof tensorflow.AssetFileDef
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates an AssetFileDef message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.AssetFileDef
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.AssetFileDef} AssetFileDef
             */
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
    
            /**
             * Creates a plain object from an AssetFileDef message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.AssetFileDef
             * @static
             * @param {tensorflow.AssetFileDef} message AssetFileDef
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this AssetFileDef to JSON.
             * @function toJSON
             * @memberof tensorflow.AssetFileDef
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            AssetFileDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return AssetFileDef;
        })();
    
        tensorflow.SaverDef = (function() {
    
            /**
             * Properties of a SaverDef.
             * @memberof tensorflow
             * @interface ISaverDef
             * @property {string|null} [filenameTensorName] SaverDef filenameTensorName
             * @property {string|null} [saveTensorName] SaverDef saveTensorName
             * @property {string|null} [restoreOpName] SaverDef restoreOpName
             * @property {number|null} [maxToKeep] SaverDef maxToKeep
             * @property {boolean|null} [sharded] SaverDef sharded
             * @property {number|null} [keepCheckpointEveryNHours] SaverDef keepCheckpointEveryNHours
             * @property {tensorflow.SaverDef.CheckpointFormatVersion|null} [version] SaverDef version
             */
    
            /**
             * Constructs a new SaverDef.
             * @memberof tensorflow
             * @classdesc Represents a SaverDef.
             * @implements ISaverDef
             * @constructor
             * @param {tensorflow.ISaverDef=} [properties] Properties to set
             */
            function SaverDef(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * SaverDef filenameTensorName.
             * @member {string} filenameTensorName
             * @memberof tensorflow.SaverDef
             * @instance
             */
            SaverDef.prototype.filenameTensorName = "";
    
            /**
             * SaverDef saveTensorName.
             * @member {string} saveTensorName
             * @memberof tensorflow.SaverDef
             * @instance
             */
            SaverDef.prototype.saveTensorName = "";
    
            /**
             * SaverDef restoreOpName.
             * @member {string} restoreOpName
             * @memberof tensorflow.SaverDef
             * @instance
             */
            SaverDef.prototype.restoreOpName = "";
    
            /**
             * SaverDef maxToKeep.
             * @member {number} maxToKeep
             * @memberof tensorflow.SaverDef
             * @instance
             */
            SaverDef.prototype.maxToKeep = 0;
    
            /**
             * SaverDef sharded.
             * @member {boolean} sharded
             * @memberof tensorflow.SaverDef
             * @instance
             */
            SaverDef.prototype.sharded = false;
    
            /**
             * SaverDef keepCheckpointEveryNHours.
             * @member {number} keepCheckpointEveryNHours
             * @memberof tensorflow.SaverDef
             * @instance
             */
            SaverDef.prototype.keepCheckpointEveryNHours = 0;
    
            /**
             * SaverDef version.
             * @member {tensorflow.SaverDef.CheckpointFormatVersion} version
             * @memberof tensorflow.SaverDef
             * @instance
             */
            SaverDef.prototype.version = 0;
    
            /**
             * Creates a new SaverDef instance using the specified properties.
             * @function create
             * @memberof tensorflow.SaverDef
             * @static
             * @param {tensorflow.ISaverDef=} [properties] Properties to set
             * @returns {tensorflow.SaverDef} SaverDef instance
             */
            SaverDef.create = function create(properties) {
                return new SaverDef(properties);
            };
    
            /**
             * Encodes the specified SaverDef message. Does not implicitly {@link tensorflow.SaverDef.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.SaverDef
             * @static
             * @param {tensorflow.ISaverDef} message SaverDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            SaverDef.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.filenameTensorName != null && message.hasOwnProperty("filenameTensorName"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.filenameTensorName);
                if (message.saveTensorName != null && message.hasOwnProperty("saveTensorName"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.saveTensorName);
                if (message.restoreOpName != null && message.hasOwnProperty("restoreOpName"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.restoreOpName);
                if (message.maxToKeep != null && message.hasOwnProperty("maxToKeep"))
                    writer.uint32(/* id 4, wireType 0 =*/32).int32(message.maxToKeep);
                if (message.sharded != null && message.hasOwnProperty("sharded"))
                    writer.uint32(/* id 5, wireType 0 =*/40).bool(message.sharded);
                if (message.keepCheckpointEveryNHours != null && message.hasOwnProperty("keepCheckpointEveryNHours"))
                    writer.uint32(/* id 6, wireType 5 =*/53).float(message.keepCheckpointEveryNHours);
                if (message.version != null && message.hasOwnProperty("version"))
                    writer.uint32(/* id 7, wireType 0 =*/56).int32(message.version);
                return writer;
            };
    
            /**
             * Encodes the specified SaverDef message, length delimited. Does not implicitly {@link tensorflow.SaverDef.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.SaverDef
             * @static
             * @param {tensorflow.ISaverDef} message SaverDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            SaverDef.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a SaverDef message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.SaverDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.SaverDef} SaverDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a SaverDef message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.SaverDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.SaverDef} SaverDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            SaverDef.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a SaverDef message.
             * @function verify
             * @memberof tensorflow.SaverDef
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a SaverDef message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.SaverDef
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.SaverDef} SaverDef
             */
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
    
            /**
             * Creates a plain object from a SaverDef message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.SaverDef
             * @static
             * @param {tensorflow.SaverDef} message SaverDef
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this SaverDef to JSON.
             * @function toJSON
             * @memberof tensorflow.SaverDef
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            SaverDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            /**
             * CheckpointFormatVersion enum.
             * @name tensorflow.SaverDef.CheckpointFormatVersion
             * @enum {string}
             * @property {number} LEGACY=0 LEGACY value
             * @property {number} V1=1 V1 value
             * @property {number} V2=2 V2 value
             */
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
    
            /**
             * Properties of a GraphDef.
             * @memberof tensorflow
             * @interface IGraphDef
             * @property {Array.<tensorflow.INodeDef>|null} [node] GraphDef node
             * @property {tensorflow.IVersionDef|null} [versions] GraphDef versions
             * @property {number|null} [version] GraphDef version
             * @property {tensorflow.IFunctionDefLibrary|null} [library] GraphDef library
             */
    
            /**
             * Constructs a new GraphDef.
             * @memberof tensorflow
             * @classdesc Represents a GraphDef.
             * @implements IGraphDef
             * @constructor
             * @param {tensorflow.IGraphDef=} [properties] Properties to set
             */
            function GraphDef(properties) {
                this.node = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * GraphDef node.
             * @member {Array.<tensorflow.INodeDef>} node
             * @memberof tensorflow.GraphDef
             * @instance
             */
            GraphDef.prototype.node = $util.emptyArray;
    
            /**
             * GraphDef versions.
             * @member {tensorflow.IVersionDef|null|undefined} versions
             * @memberof tensorflow.GraphDef
             * @instance
             */
            GraphDef.prototype.versions = null;
    
            /**
             * GraphDef version.
             * @member {number} version
             * @memberof tensorflow.GraphDef
             * @instance
             */
            GraphDef.prototype.version = 0;
    
            /**
             * GraphDef library.
             * @member {tensorflow.IFunctionDefLibrary|null|undefined} library
             * @memberof tensorflow.GraphDef
             * @instance
             */
            GraphDef.prototype.library = null;
    
            /**
             * Creates a new GraphDef instance using the specified properties.
             * @function create
             * @memberof tensorflow.GraphDef
             * @static
             * @param {tensorflow.IGraphDef=} [properties] Properties to set
             * @returns {tensorflow.GraphDef} GraphDef instance
             */
            GraphDef.create = function create(properties) {
                return new GraphDef(properties);
            };
    
            /**
             * Encodes the specified GraphDef message. Does not implicitly {@link tensorflow.GraphDef.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.GraphDef
             * @static
             * @param {tensorflow.IGraphDef} message GraphDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            GraphDef.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.node != null && message.node.length)
                    for (var i = 0; i < message.node.length; ++i)
                        $root.tensorflow.NodeDef.encode(message.node[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                if (message.library != null && message.hasOwnProperty("library"))
                    $root.tensorflow.FunctionDefLibrary.encode(message.library, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                if (message.version != null && message.hasOwnProperty("version"))
                    writer.uint32(/* id 3, wireType 0 =*/24).int32(message.version);
                if (message.versions != null && message.hasOwnProperty("versions"))
                    $root.tensorflow.VersionDef.encode(message.versions, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
                return writer;
            };
    
            /**
             * Encodes the specified GraphDef message, length delimited. Does not implicitly {@link tensorflow.GraphDef.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.GraphDef
             * @static
             * @param {tensorflow.IGraphDef} message GraphDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            GraphDef.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a GraphDef message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.GraphDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.GraphDef} GraphDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a GraphDef message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.GraphDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.GraphDef} GraphDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            GraphDef.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a GraphDef message.
             * @function verify
             * @memberof tensorflow.GraphDef
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a GraphDef message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.GraphDef
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.GraphDef} GraphDef
             */
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
    
            /**
             * Creates a plain object from a GraphDef message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.GraphDef
             * @static
             * @param {tensorflow.GraphDef} message GraphDef
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this GraphDef to JSON.
             * @function toJSON
             * @memberof tensorflow.GraphDef
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            GraphDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return GraphDef;
        })();
    
        tensorflow.OpDef = (function() {
    
            /**
             * Properties of an OpDef.
             * @memberof tensorflow
             * @interface IOpDef
             * @property {string|null} [name] OpDef name
             * @property {Array.<tensorflow.OpDef.IArgDef>|null} [inputArg] OpDef inputArg
             * @property {Array.<tensorflow.OpDef.IArgDef>|null} [outputArg] OpDef outputArg
             * @property {Array.<tensorflow.OpDef.IAttrDef>|null} [attr] OpDef attr
             * @property {tensorflow.IOpDeprecation|null} [deprecation] OpDef deprecation
             * @property {string|null} [summary] OpDef summary
             * @property {string|null} [description] OpDef description
             * @property {boolean|null} [isCommutative] OpDef isCommutative
             * @property {boolean|null} [isAggregate] OpDef isAggregate
             * @property {boolean|null} [isStateful] OpDef isStateful
             * @property {boolean|null} [allowsUninitializedInput] OpDef allowsUninitializedInput
             */
    
            /**
             * Constructs a new OpDef.
             * @memberof tensorflow
             * @classdesc Represents an OpDef.
             * @implements IOpDef
             * @constructor
             * @param {tensorflow.IOpDef=} [properties] Properties to set
             */
            function OpDef(properties) {
                this.inputArg = [];
                this.outputArg = [];
                this.attr = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * OpDef name.
             * @member {string} name
             * @memberof tensorflow.OpDef
             * @instance
             */
            OpDef.prototype.name = "";
    
            /**
             * OpDef inputArg.
             * @member {Array.<tensorflow.OpDef.IArgDef>} inputArg
             * @memberof tensorflow.OpDef
             * @instance
             */
            OpDef.prototype.inputArg = $util.emptyArray;
    
            /**
             * OpDef outputArg.
             * @member {Array.<tensorflow.OpDef.IArgDef>} outputArg
             * @memberof tensorflow.OpDef
             * @instance
             */
            OpDef.prototype.outputArg = $util.emptyArray;
    
            /**
             * OpDef attr.
             * @member {Array.<tensorflow.OpDef.IAttrDef>} attr
             * @memberof tensorflow.OpDef
             * @instance
             */
            OpDef.prototype.attr = $util.emptyArray;
    
            /**
             * OpDef deprecation.
             * @member {tensorflow.IOpDeprecation|null|undefined} deprecation
             * @memberof tensorflow.OpDef
             * @instance
             */
            OpDef.prototype.deprecation = null;
    
            /**
             * OpDef summary.
             * @member {string} summary
             * @memberof tensorflow.OpDef
             * @instance
             */
            OpDef.prototype.summary = "";
    
            /**
             * OpDef description.
             * @member {string} description
             * @memberof tensorflow.OpDef
             * @instance
             */
            OpDef.prototype.description = "";
    
            /**
             * OpDef isCommutative.
             * @member {boolean} isCommutative
             * @memberof tensorflow.OpDef
             * @instance
             */
            OpDef.prototype.isCommutative = false;
    
            /**
             * OpDef isAggregate.
             * @member {boolean} isAggregate
             * @memberof tensorflow.OpDef
             * @instance
             */
            OpDef.prototype.isAggregate = false;
    
            /**
             * OpDef isStateful.
             * @member {boolean} isStateful
             * @memberof tensorflow.OpDef
             * @instance
             */
            OpDef.prototype.isStateful = false;
    
            /**
             * OpDef allowsUninitializedInput.
             * @member {boolean} allowsUninitializedInput
             * @memberof tensorflow.OpDef
             * @instance
             */
            OpDef.prototype.allowsUninitializedInput = false;
    
            /**
             * Creates a new OpDef instance using the specified properties.
             * @function create
             * @memberof tensorflow.OpDef
             * @static
             * @param {tensorflow.IOpDef=} [properties] Properties to set
             * @returns {tensorflow.OpDef} OpDef instance
             */
            OpDef.create = function create(properties) {
                return new OpDef(properties);
            };
    
            /**
             * Encodes the specified OpDef message. Does not implicitly {@link tensorflow.OpDef.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.OpDef
             * @static
             * @param {tensorflow.IOpDef} message OpDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            OpDef.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.name != null && message.hasOwnProperty("name"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
                if (message.inputArg != null && message.inputArg.length)
                    for (var i = 0; i < message.inputArg.length; ++i)
                        $root.tensorflow.OpDef.ArgDef.encode(message.inputArg[i], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                if (message.outputArg != null && message.outputArg.length)
                    for (var i = 0; i < message.outputArg.length; ++i)
                        $root.tensorflow.OpDef.ArgDef.encode(message.outputArg[i], writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
                if (message.attr != null && message.attr.length)
                    for (var i = 0; i < message.attr.length; ++i)
                        $root.tensorflow.OpDef.AttrDef.encode(message.attr[i], writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
                if (message.summary != null && message.hasOwnProperty("summary"))
                    writer.uint32(/* id 5, wireType 2 =*/42).string(message.summary);
                if (message.description != null && message.hasOwnProperty("description"))
                    writer.uint32(/* id 6, wireType 2 =*/50).string(message.description);
                if (message.deprecation != null && message.hasOwnProperty("deprecation"))
                    $root.tensorflow.OpDeprecation.encode(message.deprecation, writer.uint32(/* id 8, wireType 2 =*/66).fork()).ldelim();
                if (message.isAggregate != null && message.hasOwnProperty("isAggregate"))
                    writer.uint32(/* id 16, wireType 0 =*/128).bool(message.isAggregate);
                if (message.isStateful != null && message.hasOwnProperty("isStateful"))
                    writer.uint32(/* id 17, wireType 0 =*/136).bool(message.isStateful);
                if (message.isCommutative != null && message.hasOwnProperty("isCommutative"))
                    writer.uint32(/* id 18, wireType 0 =*/144).bool(message.isCommutative);
                if (message.allowsUninitializedInput != null && message.hasOwnProperty("allowsUninitializedInput"))
                    writer.uint32(/* id 19, wireType 0 =*/152).bool(message.allowsUninitializedInput);
                return writer;
            };
    
            /**
             * Encodes the specified OpDef message, length delimited. Does not implicitly {@link tensorflow.OpDef.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.OpDef
             * @static
             * @param {tensorflow.IOpDef} message OpDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            OpDef.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes an OpDef message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.OpDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.OpDef} OpDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes an OpDef message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.OpDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.OpDef} OpDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            OpDef.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies an OpDef message.
             * @function verify
             * @memberof tensorflow.OpDef
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates an OpDef message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.OpDef
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.OpDef} OpDef
             */
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
    
            /**
             * Creates a plain object from an OpDef message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.OpDef
             * @static
             * @param {tensorflow.OpDef} message OpDef
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this OpDef to JSON.
             * @function toJSON
             * @memberof tensorflow.OpDef
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            OpDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            OpDef.ArgDef = (function() {
    
                /**
                 * Properties of an ArgDef.
                 * @memberof tensorflow.OpDef
                 * @interface IArgDef
                 * @property {string|null} [name] ArgDef name
                 * @property {string|null} [description] ArgDef description
                 * @property {tensorflow.DataType|null} [type] ArgDef type
                 * @property {string|null} [typeAttr] ArgDef typeAttr
                 * @property {string|null} [numberAttr] ArgDef numberAttr
                 * @property {string|null} [typeListAttr] ArgDef typeListAttr
                 * @property {boolean|null} [isRef] ArgDef isRef
                 */
    
                /**
                 * Constructs a new ArgDef.
                 * @memberof tensorflow.OpDef
                 * @classdesc Represents an ArgDef.
                 * @implements IArgDef
                 * @constructor
                 * @param {tensorflow.OpDef.IArgDef=} [properties] Properties to set
                 */
                function ArgDef(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * ArgDef name.
                 * @member {string} name
                 * @memberof tensorflow.OpDef.ArgDef
                 * @instance
                 */
                ArgDef.prototype.name = "";
    
                /**
                 * ArgDef description.
                 * @member {string} description
                 * @memberof tensorflow.OpDef.ArgDef
                 * @instance
                 */
                ArgDef.prototype.description = "";
    
                /**
                 * ArgDef type.
                 * @member {tensorflow.DataType} type
                 * @memberof tensorflow.OpDef.ArgDef
                 * @instance
                 */
                ArgDef.prototype.type = 0;
    
                /**
                 * ArgDef typeAttr.
                 * @member {string} typeAttr
                 * @memberof tensorflow.OpDef.ArgDef
                 * @instance
                 */
                ArgDef.prototype.typeAttr = "";
    
                /**
                 * ArgDef numberAttr.
                 * @member {string} numberAttr
                 * @memberof tensorflow.OpDef.ArgDef
                 * @instance
                 */
                ArgDef.prototype.numberAttr = "";
    
                /**
                 * ArgDef typeListAttr.
                 * @member {string} typeListAttr
                 * @memberof tensorflow.OpDef.ArgDef
                 * @instance
                 */
                ArgDef.prototype.typeListAttr = "";
    
                /**
                 * ArgDef isRef.
                 * @member {boolean} isRef
                 * @memberof tensorflow.OpDef.ArgDef
                 * @instance
                 */
                ArgDef.prototype.isRef = false;
    
                /**
                 * Creates a new ArgDef instance using the specified properties.
                 * @function create
                 * @memberof tensorflow.OpDef.ArgDef
                 * @static
                 * @param {tensorflow.OpDef.IArgDef=} [properties] Properties to set
                 * @returns {tensorflow.OpDef.ArgDef} ArgDef instance
                 */
                ArgDef.create = function create(properties) {
                    return new ArgDef(properties);
                };
    
                /**
                 * Encodes the specified ArgDef message. Does not implicitly {@link tensorflow.OpDef.ArgDef.verify|verify} messages.
                 * @function encode
                 * @memberof tensorflow.OpDef.ArgDef
                 * @static
                 * @param {tensorflow.OpDef.IArgDef} message ArgDef message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                ArgDef.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.name != null && message.hasOwnProperty("name"))
                        writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
                    if (message.description != null && message.hasOwnProperty("description"))
                        writer.uint32(/* id 2, wireType 2 =*/18).string(message.description);
                    if (message.type != null && message.hasOwnProperty("type"))
                        writer.uint32(/* id 3, wireType 0 =*/24).int32(message.type);
                    if (message.typeAttr != null && message.hasOwnProperty("typeAttr"))
                        writer.uint32(/* id 4, wireType 2 =*/34).string(message.typeAttr);
                    if (message.numberAttr != null && message.hasOwnProperty("numberAttr"))
                        writer.uint32(/* id 5, wireType 2 =*/42).string(message.numberAttr);
                    if (message.typeListAttr != null && message.hasOwnProperty("typeListAttr"))
                        writer.uint32(/* id 6, wireType 2 =*/50).string(message.typeListAttr);
                    if (message.isRef != null && message.hasOwnProperty("isRef"))
                        writer.uint32(/* id 16, wireType 0 =*/128).bool(message.isRef);
                    return writer;
                };
    
                /**
                 * Encodes the specified ArgDef message, length delimited. Does not implicitly {@link tensorflow.OpDef.ArgDef.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof tensorflow.OpDef.ArgDef
                 * @static
                 * @param {tensorflow.OpDef.IArgDef} message ArgDef message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                ArgDef.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes an ArgDef message from the specified reader or buffer.
                 * @function decode
                 * @memberof tensorflow.OpDef.ArgDef
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {tensorflow.OpDef.ArgDef} ArgDef
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
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
    
                /**
                 * Decodes an ArgDef message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof tensorflow.OpDef.ArgDef
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {tensorflow.OpDef.ArgDef} ArgDef
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                ArgDef.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies an ArgDef message.
                 * @function verify
                 * @memberof tensorflow.OpDef.ArgDef
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
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
    
                /**
                 * Creates an ArgDef message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof tensorflow.OpDef.ArgDef
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {tensorflow.OpDef.ArgDef} ArgDef
                 */
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
    
                /**
                 * Creates a plain object from an ArgDef message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof tensorflow.OpDef.ArgDef
                 * @static
                 * @param {tensorflow.OpDef.ArgDef} message ArgDef
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
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
    
                /**
                 * Converts this ArgDef to JSON.
                 * @function toJSON
                 * @memberof tensorflow.OpDef.ArgDef
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                ArgDef.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return ArgDef;
            })();
    
            OpDef.AttrDef = (function() {
    
                /**
                 * Properties of an AttrDef.
                 * @memberof tensorflow.OpDef
                 * @interface IAttrDef
                 * @property {string|null} [name] AttrDef name
                 * @property {string|null} [type] AttrDef type
                 * @property {tensorflow.IAttrValue|null} [defaultValue] AttrDef defaultValue
                 * @property {string|null} [description] AttrDef description
                 * @property {boolean|null} [hasMinimum] AttrDef hasMinimum
                 * @property {number|Long|null} [minimum] AttrDef minimum
                 * @property {tensorflow.IAttrValue|null} [allowedValues] AttrDef allowedValues
                 */
    
                /**
                 * Constructs a new AttrDef.
                 * @memberof tensorflow.OpDef
                 * @classdesc Represents an AttrDef.
                 * @implements IAttrDef
                 * @constructor
                 * @param {tensorflow.OpDef.IAttrDef=} [properties] Properties to set
                 */
                function AttrDef(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * AttrDef name.
                 * @member {string} name
                 * @memberof tensorflow.OpDef.AttrDef
                 * @instance
                 */
                AttrDef.prototype.name = "";
    
                /**
                 * AttrDef type.
                 * @member {string} type
                 * @memberof tensorflow.OpDef.AttrDef
                 * @instance
                 */
                AttrDef.prototype.type = "";
    
                /**
                 * AttrDef defaultValue.
                 * @member {tensorflow.IAttrValue|null|undefined} defaultValue
                 * @memberof tensorflow.OpDef.AttrDef
                 * @instance
                 */
                AttrDef.prototype.defaultValue = null;
    
                /**
                 * AttrDef description.
                 * @member {string} description
                 * @memberof tensorflow.OpDef.AttrDef
                 * @instance
                 */
                AttrDef.prototype.description = "";
    
                /**
                 * AttrDef hasMinimum.
                 * @member {boolean} hasMinimum
                 * @memberof tensorflow.OpDef.AttrDef
                 * @instance
                 */
                AttrDef.prototype.hasMinimum = false;
    
                /**
                 * AttrDef minimum.
                 * @member {number|Long} minimum
                 * @memberof tensorflow.OpDef.AttrDef
                 * @instance
                 */
                AttrDef.prototype.minimum = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                /**
                 * AttrDef allowedValues.
                 * @member {tensorflow.IAttrValue|null|undefined} allowedValues
                 * @memberof tensorflow.OpDef.AttrDef
                 * @instance
                 */
                AttrDef.prototype.allowedValues = null;
    
                /**
                 * Creates a new AttrDef instance using the specified properties.
                 * @function create
                 * @memberof tensorflow.OpDef.AttrDef
                 * @static
                 * @param {tensorflow.OpDef.IAttrDef=} [properties] Properties to set
                 * @returns {tensorflow.OpDef.AttrDef} AttrDef instance
                 */
                AttrDef.create = function create(properties) {
                    return new AttrDef(properties);
                };
    
                /**
                 * Encodes the specified AttrDef message. Does not implicitly {@link tensorflow.OpDef.AttrDef.verify|verify} messages.
                 * @function encode
                 * @memberof tensorflow.OpDef.AttrDef
                 * @static
                 * @param {tensorflow.OpDef.IAttrDef} message AttrDef message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                AttrDef.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.name != null && message.hasOwnProperty("name"))
                        writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
                    if (message.type != null && message.hasOwnProperty("type"))
                        writer.uint32(/* id 2, wireType 2 =*/18).string(message.type);
                    if (message.defaultValue != null && message.hasOwnProperty("defaultValue"))
                        $root.tensorflow.AttrValue.encode(message.defaultValue, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
                    if (message.description != null && message.hasOwnProperty("description"))
                        writer.uint32(/* id 4, wireType 2 =*/34).string(message.description);
                    if (message.hasMinimum != null && message.hasOwnProperty("hasMinimum"))
                        writer.uint32(/* id 5, wireType 0 =*/40).bool(message.hasMinimum);
                    if (message.minimum != null && message.hasOwnProperty("minimum"))
                        writer.uint32(/* id 6, wireType 0 =*/48).int64(message.minimum);
                    if (message.allowedValues != null && message.hasOwnProperty("allowedValues"))
                        $root.tensorflow.AttrValue.encode(message.allowedValues, writer.uint32(/* id 7, wireType 2 =*/58).fork()).ldelim();
                    return writer;
                };
    
                /**
                 * Encodes the specified AttrDef message, length delimited. Does not implicitly {@link tensorflow.OpDef.AttrDef.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof tensorflow.OpDef.AttrDef
                 * @static
                 * @param {tensorflow.OpDef.IAttrDef} message AttrDef message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                AttrDef.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes an AttrDef message from the specified reader or buffer.
                 * @function decode
                 * @memberof tensorflow.OpDef.AttrDef
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {tensorflow.OpDef.AttrDef} AttrDef
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
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
    
                /**
                 * Decodes an AttrDef message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof tensorflow.OpDef.AttrDef
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {tensorflow.OpDef.AttrDef} AttrDef
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                AttrDef.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies an AttrDef message.
                 * @function verify
                 * @memberof tensorflow.OpDef.AttrDef
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
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
    
                /**
                 * Creates an AttrDef message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof tensorflow.OpDef.AttrDef
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {tensorflow.OpDef.AttrDef} AttrDef
                 */
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
    
                /**
                 * Creates a plain object from an AttrDef message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof tensorflow.OpDef.AttrDef
                 * @static
                 * @param {tensorflow.OpDef.AttrDef} message AttrDef
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
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
    
                /**
                 * Converts this AttrDef to JSON.
                 * @function toJSON
                 * @memberof tensorflow.OpDef.AttrDef
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                AttrDef.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return AttrDef;
            })();
    
            return OpDef;
        })();
    
        tensorflow.OpDeprecation = (function() {
    
            /**
             * Properties of an OpDeprecation.
             * @memberof tensorflow
             * @interface IOpDeprecation
             * @property {number|null} [version] OpDeprecation version
             * @property {string|null} [explanation] OpDeprecation explanation
             */
    
            /**
             * Constructs a new OpDeprecation.
             * @memberof tensorflow
             * @classdesc Represents an OpDeprecation.
             * @implements IOpDeprecation
             * @constructor
             * @param {tensorflow.IOpDeprecation=} [properties] Properties to set
             */
            function OpDeprecation(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * OpDeprecation version.
             * @member {number} version
             * @memberof tensorflow.OpDeprecation
             * @instance
             */
            OpDeprecation.prototype.version = 0;
    
            /**
             * OpDeprecation explanation.
             * @member {string} explanation
             * @memberof tensorflow.OpDeprecation
             * @instance
             */
            OpDeprecation.prototype.explanation = "";
    
            /**
             * Creates a new OpDeprecation instance using the specified properties.
             * @function create
             * @memberof tensorflow.OpDeprecation
             * @static
             * @param {tensorflow.IOpDeprecation=} [properties] Properties to set
             * @returns {tensorflow.OpDeprecation} OpDeprecation instance
             */
            OpDeprecation.create = function create(properties) {
                return new OpDeprecation(properties);
            };
    
            /**
             * Encodes the specified OpDeprecation message. Does not implicitly {@link tensorflow.OpDeprecation.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.OpDeprecation
             * @static
             * @param {tensorflow.IOpDeprecation} message OpDeprecation message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            OpDeprecation.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.version != null && message.hasOwnProperty("version"))
                    writer.uint32(/* id 1, wireType 0 =*/8).int32(message.version);
                if (message.explanation != null && message.hasOwnProperty("explanation"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.explanation);
                return writer;
            };
    
            /**
             * Encodes the specified OpDeprecation message, length delimited. Does not implicitly {@link tensorflow.OpDeprecation.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.OpDeprecation
             * @static
             * @param {tensorflow.IOpDeprecation} message OpDeprecation message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            OpDeprecation.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes an OpDeprecation message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.OpDeprecation
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.OpDeprecation} OpDeprecation
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes an OpDeprecation message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.OpDeprecation
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.OpDeprecation} OpDeprecation
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            OpDeprecation.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies an OpDeprecation message.
             * @function verify
             * @memberof tensorflow.OpDeprecation
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates an OpDeprecation message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.OpDeprecation
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.OpDeprecation} OpDeprecation
             */
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
    
            /**
             * Creates a plain object from an OpDeprecation message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.OpDeprecation
             * @static
             * @param {tensorflow.OpDeprecation} message OpDeprecation
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this OpDeprecation to JSON.
             * @function toJSON
             * @memberof tensorflow.OpDeprecation
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            OpDeprecation.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return OpDeprecation;
        })();
    
        tensorflow.OpList = (function() {
    
            /**
             * Properties of an OpList.
             * @memberof tensorflow
             * @interface IOpList
             * @property {Array.<tensorflow.IOpDef>|null} [op] OpList op
             */
    
            /**
             * Constructs a new OpList.
             * @memberof tensorflow
             * @classdesc Represents an OpList.
             * @implements IOpList
             * @constructor
             * @param {tensorflow.IOpList=} [properties] Properties to set
             */
            function OpList(properties) {
                this.op = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * OpList op.
             * @member {Array.<tensorflow.IOpDef>} op
             * @memberof tensorflow.OpList
             * @instance
             */
            OpList.prototype.op = $util.emptyArray;
    
            /**
             * Creates a new OpList instance using the specified properties.
             * @function create
             * @memberof tensorflow.OpList
             * @static
             * @param {tensorflow.IOpList=} [properties] Properties to set
             * @returns {tensorflow.OpList} OpList instance
             */
            OpList.create = function create(properties) {
                return new OpList(properties);
            };
    
            /**
             * Encodes the specified OpList message. Does not implicitly {@link tensorflow.OpList.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.OpList
             * @static
             * @param {tensorflow.IOpList} message OpList message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            OpList.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.op != null && message.op.length)
                    for (var i = 0; i < message.op.length; ++i)
                        $root.tensorflow.OpDef.encode(message.op[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                return writer;
            };
    
            /**
             * Encodes the specified OpList message, length delimited. Does not implicitly {@link tensorflow.OpList.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.OpList
             * @static
             * @param {tensorflow.IOpList} message OpList message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            OpList.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes an OpList message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.OpList
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.OpList} OpList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes an OpList message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.OpList
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.OpList} OpList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            OpList.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies an OpList message.
             * @function verify
             * @memberof tensorflow.OpList
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates an OpList message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.OpList
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.OpList} OpList
             */
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
    
            /**
             * Creates a plain object from an OpList message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.OpList
             * @static
             * @param {tensorflow.OpList} message OpList
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this OpList to JSON.
             * @function toJSON
             * @memberof tensorflow.OpList
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            OpList.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return OpList;
        })();
    
        tensorflow.TensorShapeProto = (function() {
    
            /**
             * Properties of a TensorShapeProto.
             * @memberof tensorflow
             * @interface ITensorShapeProto
             * @property {Array.<tensorflow.TensorShapeProto.IDim>|null} [dim] TensorShapeProto dim
             * @property {boolean|null} [unknownRank] TensorShapeProto unknownRank
             */
    
            /**
             * Constructs a new TensorShapeProto.
             * @memberof tensorflow
             * @classdesc Represents a TensorShapeProto.
             * @implements ITensorShapeProto
             * @constructor
             * @param {tensorflow.ITensorShapeProto=} [properties] Properties to set
             */
            function TensorShapeProto(properties) {
                this.dim = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * TensorShapeProto dim.
             * @member {Array.<tensorflow.TensorShapeProto.IDim>} dim
             * @memberof tensorflow.TensorShapeProto
             * @instance
             */
            TensorShapeProto.prototype.dim = $util.emptyArray;
    
            /**
             * TensorShapeProto unknownRank.
             * @member {boolean} unknownRank
             * @memberof tensorflow.TensorShapeProto
             * @instance
             */
            TensorShapeProto.prototype.unknownRank = false;
    
            /**
             * Creates a new TensorShapeProto instance using the specified properties.
             * @function create
             * @memberof tensorflow.TensorShapeProto
             * @static
             * @param {tensorflow.ITensorShapeProto=} [properties] Properties to set
             * @returns {tensorflow.TensorShapeProto} TensorShapeProto instance
             */
            TensorShapeProto.create = function create(properties) {
                return new TensorShapeProto(properties);
            };
    
            /**
             * Encodes the specified TensorShapeProto message. Does not implicitly {@link tensorflow.TensorShapeProto.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.TensorShapeProto
             * @static
             * @param {tensorflow.ITensorShapeProto} message TensorShapeProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TensorShapeProto.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.dim != null && message.dim.length)
                    for (var i = 0; i < message.dim.length; ++i)
                        $root.tensorflow.TensorShapeProto.Dim.encode(message.dim[i], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                if (message.unknownRank != null && message.hasOwnProperty("unknownRank"))
                    writer.uint32(/* id 3, wireType 0 =*/24).bool(message.unknownRank);
                return writer;
            };
    
            /**
             * Encodes the specified TensorShapeProto message, length delimited. Does not implicitly {@link tensorflow.TensorShapeProto.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.TensorShapeProto
             * @static
             * @param {tensorflow.ITensorShapeProto} message TensorShapeProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TensorShapeProto.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a TensorShapeProto message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.TensorShapeProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.TensorShapeProto} TensorShapeProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a TensorShapeProto message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.TensorShapeProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.TensorShapeProto} TensorShapeProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            TensorShapeProto.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a TensorShapeProto message.
             * @function verify
             * @memberof tensorflow.TensorShapeProto
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a TensorShapeProto message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.TensorShapeProto
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.TensorShapeProto} TensorShapeProto
             */
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
    
            /**
             * Creates a plain object from a TensorShapeProto message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.TensorShapeProto
             * @static
             * @param {tensorflow.TensorShapeProto} message TensorShapeProto
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this TensorShapeProto to JSON.
             * @function toJSON
             * @memberof tensorflow.TensorShapeProto
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            TensorShapeProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            TensorShapeProto.Dim = (function() {
    
                /**
                 * Properties of a Dim.
                 * @memberof tensorflow.TensorShapeProto
                 * @interface IDim
                 * @property {number|Long|null} [size] Dim size
                 * @property {string|null} [name] Dim name
                 */
    
                /**
                 * Constructs a new Dim.
                 * @memberof tensorflow.TensorShapeProto
                 * @classdesc Represents a Dim.
                 * @implements IDim
                 * @constructor
                 * @param {tensorflow.TensorShapeProto.IDim=} [properties] Properties to set
                 */
                function Dim(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * Dim size.
                 * @member {number|Long} size
                 * @memberof tensorflow.TensorShapeProto.Dim
                 * @instance
                 */
                Dim.prototype.size = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                /**
                 * Dim name.
                 * @member {string} name
                 * @memberof tensorflow.TensorShapeProto.Dim
                 * @instance
                 */
                Dim.prototype.name = "";
    
                /**
                 * Creates a new Dim instance using the specified properties.
                 * @function create
                 * @memberof tensorflow.TensorShapeProto.Dim
                 * @static
                 * @param {tensorflow.TensorShapeProto.IDim=} [properties] Properties to set
                 * @returns {tensorflow.TensorShapeProto.Dim} Dim instance
                 */
                Dim.create = function create(properties) {
                    return new Dim(properties);
                };
    
                /**
                 * Encodes the specified Dim message. Does not implicitly {@link tensorflow.TensorShapeProto.Dim.verify|verify} messages.
                 * @function encode
                 * @memberof tensorflow.TensorShapeProto.Dim
                 * @static
                 * @param {tensorflow.TensorShapeProto.IDim} message Dim message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Dim.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.size != null && message.hasOwnProperty("size"))
                        writer.uint32(/* id 1, wireType 0 =*/8).int64(message.size);
                    if (message.name != null && message.hasOwnProperty("name"))
                        writer.uint32(/* id 2, wireType 2 =*/18).string(message.name);
                    return writer;
                };
    
                /**
                 * Encodes the specified Dim message, length delimited. Does not implicitly {@link tensorflow.TensorShapeProto.Dim.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof tensorflow.TensorShapeProto.Dim
                 * @static
                 * @param {tensorflow.TensorShapeProto.IDim} message Dim message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Dim.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes a Dim message from the specified reader or buffer.
                 * @function decode
                 * @memberof tensorflow.TensorShapeProto.Dim
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {tensorflow.TensorShapeProto.Dim} Dim
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
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
    
                /**
                 * Decodes a Dim message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof tensorflow.TensorShapeProto.Dim
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {tensorflow.TensorShapeProto.Dim} Dim
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                Dim.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies a Dim message.
                 * @function verify
                 * @memberof tensorflow.TensorShapeProto.Dim
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
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
    
                /**
                 * Creates a Dim message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof tensorflow.TensorShapeProto.Dim
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {tensorflow.TensorShapeProto.Dim} Dim
                 */
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
    
                /**
                 * Creates a plain object from a Dim message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof tensorflow.TensorShapeProto.Dim
                 * @static
                 * @param {tensorflow.TensorShapeProto.Dim} message Dim
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
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
    
                /**
                 * Converts this Dim to JSON.
                 * @function toJSON
                 * @memberof tensorflow.TensorShapeProto.Dim
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                Dim.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return Dim;
            })();
    
            return TensorShapeProto;
        })();
    
        /**
         * DataType enum.
         * @name tensorflow.DataType
         * @enum {string}
         * @property {number} DT_INVALID=0 DT_INVALID value
         * @property {number} DT_FLOAT=1 DT_FLOAT value
         * @property {number} DT_DOUBLE=2 DT_DOUBLE value
         * @property {number} DT_INT32=3 DT_INT32 value
         * @property {number} DT_UINT8=4 DT_UINT8 value
         * @property {number} DT_INT16=5 DT_INT16 value
         * @property {number} DT_INT8=6 DT_INT8 value
         * @property {number} DT_STRING=7 DT_STRING value
         * @property {number} DT_COMPLEX64=8 DT_COMPLEX64 value
         * @property {number} DT_INT64=9 DT_INT64 value
         * @property {number} DT_BOOL=10 DT_BOOL value
         * @property {number} DT_QINT8=11 DT_QINT8 value
         * @property {number} DT_QUINT8=12 DT_QUINT8 value
         * @property {number} DT_QINT32=13 DT_QINT32 value
         * @property {number} DT_BFLOAT16=14 DT_BFLOAT16 value
         * @property {number} DT_QINT16=15 DT_QINT16 value
         * @property {number} DT_QUINT16=16 DT_QUINT16 value
         * @property {number} DT_UINT16=17 DT_UINT16 value
         * @property {number} DT_COMPLEX128=18 DT_COMPLEX128 value
         * @property {number} DT_HALF=19 DT_HALF value
         * @property {number} DT_RESOURCE=20 DT_RESOURCE value
         * @property {number} DT_VARIANT=21 DT_VARIANT value
         * @property {number} DT_UINT32=22 DT_UINT32 value
         * @property {number} DT_UINT64=23 DT_UINT64 value
         * @property {number} DT_FLOAT_REF=101 DT_FLOAT_REF value
         * @property {number} DT_DOUBLE_REF=102 DT_DOUBLE_REF value
         * @property {number} DT_INT32_REF=103 DT_INT32_REF value
         * @property {number} DT_UINT8_REF=104 DT_UINT8_REF value
         * @property {number} DT_INT16_REF=105 DT_INT16_REF value
         * @property {number} DT_INT8_REF=106 DT_INT8_REF value
         * @property {number} DT_STRING_REF=107 DT_STRING_REF value
         * @property {number} DT_COMPLEX64_REF=108 DT_COMPLEX64_REF value
         * @property {number} DT_INT64_REF=109 DT_INT64_REF value
         * @property {number} DT_BOOL_REF=110 DT_BOOL_REF value
         * @property {number} DT_QINT8_REF=111 DT_QINT8_REF value
         * @property {number} DT_QUINT8_REF=112 DT_QUINT8_REF value
         * @property {number} DT_QINT32_REF=113 DT_QINT32_REF value
         * @property {number} DT_BFLOAT16_REF=114 DT_BFLOAT16_REF value
         * @property {number} DT_QINT16_REF=115 DT_QINT16_REF value
         * @property {number} DT_QUINT16_REF=116 DT_QUINT16_REF value
         * @property {number} DT_UINT16_REF=117 DT_UINT16_REF value
         * @property {number} DT_COMPLEX128_REF=118 DT_COMPLEX128_REF value
         * @property {number} DT_HALF_REF=119 DT_HALF_REF value
         * @property {number} DT_RESOURCE_REF=120 DT_RESOURCE_REF value
         * @property {number} DT_VARIANT_REF=121 DT_VARIANT_REF value
         * @property {number} DT_UINT32_REF=122 DT_UINT32_REF value
         * @property {number} DT_UINT64_REF=123 DT_UINT64_REF value
         */
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
    
            /**
             * Properties of a NodeDef.
             * @memberof tensorflow
             * @interface INodeDef
             * @property {string|null} [name] NodeDef name
             * @property {string|null} [op] NodeDef op
             * @property {Array.<string>|null} [input] NodeDef input
             * @property {string|null} [device] NodeDef device
             * @property {Object.<string,tensorflow.IAttrValue>|null} [attr] NodeDef attr
             */
    
            /**
             * Constructs a new NodeDef.
             * @memberof tensorflow
             * @classdesc Represents a NodeDef.
             * @implements INodeDef
             * @constructor
             * @param {tensorflow.INodeDef=} [properties] Properties to set
             */
            function NodeDef(properties) {
                this.input = [];
                this.attr = {};
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * NodeDef name.
             * @member {string} name
             * @memberof tensorflow.NodeDef
             * @instance
             */
            NodeDef.prototype.name = "";
    
            /**
             * NodeDef op.
             * @member {string} op
             * @memberof tensorflow.NodeDef
             * @instance
             */
            NodeDef.prototype.op = "";
    
            /**
             * NodeDef input.
             * @member {Array.<string>} input
             * @memberof tensorflow.NodeDef
             * @instance
             */
            NodeDef.prototype.input = $util.emptyArray;
    
            /**
             * NodeDef device.
             * @member {string} device
             * @memberof tensorflow.NodeDef
             * @instance
             */
            NodeDef.prototype.device = "";
    
            /**
             * NodeDef attr.
             * @member {Object.<string,tensorflow.IAttrValue>} attr
             * @memberof tensorflow.NodeDef
             * @instance
             */
            NodeDef.prototype.attr = $util.emptyObject;
    
            /**
             * Creates a new NodeDef instance using the specified properties.
             * @function create
             * @memberof tensorflow.NodeDef
             * @static
             * @param {tensorflow.INodeDef=} [properties] Properties to set
             * @returns {tensorflow.NodeDef} NodeDef instance
             */
            NodeDef.create = function create(properties) {
                return new NodeDef(properties);
            };
    
            /**
             * Encodes the specified NodeDef message. Does not implicitly {@link tensorflow.NodeDef.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.NodeDef
             * @static
             * @param {tensorflow.INodeDef} message NodeDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            NodeDef.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.name != null && message.hasOwnProperty("name"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
                if (message.op != null && message.hasOwnProperty("op"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.op);
                if (message.input != null && message.input.length)
                    for (var i = 0; i < message.input.length; ++i)
                        writer.uint32(/* id 3, wireType 2 =*/26).string(message.input[i]);
                if (message.device != null && message.hasOwnProperty("device"))
                    writer.uint32(/* id 4, wireType 2 =*/34).string(message.device);
                if (message.attr != null && message.hasOwnProperty("attr"))
                    for (var keys = Object.keys(message.attr), i = 0; i < keys.length; ++i) {
                        writer.uint32(/* id 5, wireType 2 =*/42).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]);
                        $root.tensorflow.AttrValue.encode(message.attr[keys[i]], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim().ldelim();
                    }
                return writer;
            };
    
            /**
             * Encodes the specified NodeDef message, length delimited. Does not implicitly {@link tensorflow.NodeDef.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.NodeDef
             * @static
             * @param {tensorflow.INodeDef} message NodeDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            NodeDef.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a NodeDef message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.NodeDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.NodeDef} NodeDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a NodeDef message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.NodeDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.NodeDef} NodeDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            NodeDef.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a NodeDef message.
             * @function verify
             * @memberof tensorflow.NodeDef
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a NodeDef message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.NodeDef
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.NodeDef} NodeDef
             */
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
    
            /**
             * Creates a plain object from a NodeDef message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.NodeDef
             * @static
             * @param {tensorflow.NodeDef} message NodeDef
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this NodeDef to JSON.
             * @function toJSON
             * @memberof tensorflow.NodeDef
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            NodeDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return NodeDef;
        })();
    
        tensorflow.VersionDef = (function() {
    
            /**
             * Properties of a VersionDef.
             * @memberof tensorflow
             * @interface IVersionDef
             * @property {number|null} [producer] VersionDef producer
             * @property {number|null} [minConsumer] VersionDef minConsumer
             * @property {Array.<number>|null} [badConsumers] VersionDef badConsumers
             */
    
            /**
             * Constructs a new VersionDef.
             * @memberof tensorflow
             * @classdesc Represents a VersionDef.
             * @implements IVersionDef
             * @constructor
             * @param {tensorflow.IVersionDef=} [properties] Properties to set
             */
            function VersionDef(properties) {
                this.badConsumers = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * VersionDef producer.
             * @member {number} producer
             * @memberof tensorflow.VersionDef
             * @instance
             */
            VersionDef.prototype.producer = 0;
    
            /**
             * VersionDef minConsumer.
             * @member {number} minConsumer
             * @memberof tensorflow.VersionDef
             * @instance
             */
            VersionDef.prototype.minConsumer = 0;
    
            /**
             * VersionDef badConsumers.
             * @member {Array.<number>} badConsumers
             * @memberof tensorflow.VersionDef
             * @instance
             */
            VersionDef.prototype.badConsumers = $util.emptyArray;
    
            /**
             * Creates a new VersionDef instance using the specified properties.
             * @function create
             * @memberof tensorflow.VersionDef
             * @static
             * @param {tensorflow.IVersionDef=} [properties] Properties to set
             * @returns {tensorflow.VersionDef} VersionDef instance
             */
            VersionDef.create = function create(properties) {
                return new VersionDef(properties);
            };
    
            /**
             * Encodes the specified VersionDef message. Does not implicitly {@link tensorflow.VersionDef.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.VersionDef
             * @static
             * @param {tensorflow.IVersionDef} message VersionDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            VersionDef.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.producer != null && message.hasOwnProperty("producer"))
                    writer.uint32(/* id 1, wireType 0 =*/8).int32(message.producer);
                if (message.minConsumer != null && message.hasOwnProperty("minConsumer"))
                    writer.uint32(/* id 2, wireType 0 =*/16).int32(message.minConsumer);
                if (message.badConsumers != null && message.badConsumers.length) {
                    writer.uint32(/* id 3, wireType 2 =*/26).fork();
                    for (var i = 0; i < message.badConsumers.length; ++i)
                        writer.int32(message.badConsumers[i]);
                    writer.ldelim();
                }
                return writer;
            };
    
            /**
             * Encodes the specified VersionDef message, length delimited. Does not implicitly {@link tensorflow.VersionDef.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.VersionDef
             * @static
             * @param {tensorflow.IVersionDef} message VersionDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            VersionDef.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a VersionDef message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.VersionDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.VersionDef} VersionDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a VersionDef message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.VersionDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.VersionDef} VersionDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            VersionDef.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a VersionDef message.
             * @function verify
             * @memberof tensorflow.VersionDef
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a VersionDef message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.VersionDef
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.VersionDef} VersionDef
             */
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
    
            /**
             * Creates a plain object from a VersionDef message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.VersionDef
             * @static
             * @param {tensorflow.VersionDef} message VersionDef
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this VersionDef to JSON.
             * @function toJSON
             * @memberof tensorflow.VersionDef
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            VersionDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return VersionDef;
        })();
    
        tensorflow.FunctionDefLibrary = (function() {
    
            /**
             * Properties of a FunctionDefLibrary.
             * @memberof tensorflow
             * @interface IFunctionDefLibrary
             * @property {Array.<tensorflow.IFunctionDef>|null} ["function"] FunctionDefLibrary function
             * @property {Array.<tensorflow.IGradientDef>|null} [gradient] FunctionDefLibrary gradient
             */
    
            /**
             * Constructs a new FunctionDefLibrary.
             * @memberof tensorflow
             * @classdesc Represents a FunctionDefLibrary.
             * @implements IFunctionDefLibrary
             * @constructor
             * @param {tensorflow.IFunctionDefLibrary=} [properties] Properties to set
             */
            function FunctionDefLibrary(properties) {
                this["function"] = [];
                this.gradient = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * FunctionDefLibrary function.
             * @member {Array.<tensorflow.IFunctionDef>} function
             * @memberof tensorflow.FunctionDefLibrary
             * @instance
             */
            FunctionDefLibrary.prototype["function"] = $util.emptyArray;
    
            /**
             * FunctionDefLibrary gradient.
             * @member {Array.<tensorflow.IGradientDef>} gradient
             * @memberof tensorflow.FunctionDefLibrary
             * @instance
             */
            FunctionDefLibrary.prototype.gradient = $util.emptyArray;
    
            /**
             * Creates a new FunctionDefLibrary instance using the specified properties.
             * @function create
             * @memberof tensorflow.FunctionDefLibrary
             * @static
             * @param {tensorflow.IFunctionDefLibrary=} [properties] Properties to set
             * @returns {tensorflow.FunctionDefLibrary} FunctionDefLibrary instance
             */
            FunctionDefLibrary.create = function create(properties) {
                return new FunctionDefLibrary(properties);
            };
    
            /**
             * Encodes the specified FunctionDefLibrary message. Does not implicitly {@link tensorflow.FunctionDefLibrary.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.FunctionDefLibrary
             * @static
             * @param {tensorflow.IFunctionDefLibrary} message FunctionDefLibrary message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            FunctionDefLibrary.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message["function"] != null && message["function"].length)
                    for (var i = 0; i < message["function"].length; ++i)
                        $root.tensorflow.FunctionDef.encode(message["function"][i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                if (message.gradient != null && message.gradient.length)
                    for (var i = 0; i < message.gradient.length; ++i)
                        $root.tensorflow.GradientDef.encode(message.gradient[i], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                return writer;
            };
    
            /**
             * Encodes the specified FunctionDefLibrary message, length delimited. Does not implicitly {@link tensorflow.FunctionDefLibrary.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.FunctionDefLibrary
             * @static
             * @param {tensorflow.IFunctionDefLibrary} message FunctionDefLibrary message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            FunctionDefLibrary.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a FunctionDefLibrary message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.FunctionDefLibrary
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.FunctionDefLibrary} FunctionDefLibrary
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a FunctionDefLibrary message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.FunctionDefLibrary
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.FunctionDefLibrary} FunctionDefLibrary
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            FunctionDefLibrary.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a FunctionDefLibrary message.
             * @function verify
             * @memberof tensorflow.FunctionDefLibrary
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a FunctionDefLibrary message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.FunctionDefLibrary
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.FunctionDefLibrary} FunctionDefLibrary
             */
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
    
            /**
             * Creates a plain object from a FunctionDefLibrary message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.FunctionDefLibrary
             * @static
             * @param {tensorflow.FunctionDefLibrary} message FunctionDefLibrary
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this FunctionDefLibrary to JSON.
             * @function toJSON
             * @memberof tensorflow.FunctionDefLibrary
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            FunctionDefLibrary.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return FunctionDefLibrary;
        })();
    
        tensorflow.FunctionDef = (function() {
    
            /**
             * Properties of a FunctionDef.
             * @memberof tensorflow
             * @interface IFunctionDef
             * @property {tensorflow.IOpDef|null} [signature] FunctionDef signature
             * @property {Object.<string,tensorflow.IAttrValue>|null} [attr] FunctionDef attr
             * @property {Array.<tensorflow.INodeDef>|null} [nodeDef] FunctionDef nodeDef
             * @property {Object.<string,string>|null} [ret] FunctionDef ret
             */
    
            /**
             * Constructs a new FunctionDef.
             * @memberof tensorflow
             * @classdesc Represents a FunctionDef.
             * @implements IFunctionDef
             * @constructor
             * @param {tensorflow.IFunctionDef=} [properties] Properties to set
             */
            function FunctionDef(properties) {
                this.attr = {};
                this.nodeDef = [];
                this.ret = {};
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * FunctionDef signature.
             * @member {tensorflow.IOpDef|null|undefined} signature
             * @memberof tensorflow.FunctionDef
             * @instance
             */
            FunctionDef.prototype.signature = null;
    
            /**
             * FunctionDef attr.
             * @member {Object.<string,tensorflow.IAttrValue>} attr
             * @memberof tensorflow.FunctionDef
             * @instance
             */
            FunctionDef.prototype.attr = $util.emptyObject;
    
            /**
             * FunctionDef nodeDef.
             * @member {Array.<tensorflow.INodeDef>} nodeDef
             * @memberof tensorflow.FunctionDef
             * @instance
             */
            FunctionDef.prototype.nodeDef = $util.emptyArray;
    
            /**
             * FunctionDef ret.
             * @member {Object.<string,string>} ret
             * @memberof tensorflow.FunctionDef
             * @instance
             */
            FunctionDef.prototype.ret = $util.emptyObject;
    
            /**
             * Creates a new FunctionDef instance using the specified properties.
             * @function create
             * @memberof tensorflow.FunctionDef
             * @static
             * @param {tensorflow.IFunctionDef=} [properties] Properties to set
             * @returns {tensorflow.FunctionDef} FunctionDef instance
             */
            FunctionDef.create = function create(properties) {
                return new FunctionDef(properties);
            };
    
            /**
             * Encodes the specified FunctionDef message. Does not implicitly {@link tensorflow.FunctionDef.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.FunctionDef
             * @static
             * @param {tensorflow.IFunctionDef} message FunctionDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            FunctionDef.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.signature != null && message.hasOwnProperty("signature"))
                    $root.tensorflow.OpDef.encode(message.signature, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                if (message.nodeDef != null && message.nodeDef.length)
                    for (var i = 0; i < message.nodeDef.length; ++i)
                        $root.tensorflow.NodeDef.encode(message.nodeDef[i], writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
                if (message.ret != null && message.hasOwnProperty("ret"))
                    for (var keys = Object.keys(message.ret), i = 0; i < keys.length; ++i)
                        writer.uint32(/* id 4, wireType 2 =*/34).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]).uint32(/* id 2, wireType 2 =*/18).string(message.ret[keys[i]]).ldelim();
                if (message.attr != null && message.hasOwnProperty("attr"))
                    for (var keys = Object.keys(message.attr), i = 0; i < keys.length; ++i) {
                        writer.uint32(/* id 5, wireType 2 =*/42).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]);
                        $root.tensorflow.AttrValue.encode(message.attr[keys[i]], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim().ldelim();
                    }
                return writer;
            };
    
            /**
             * Encodes the specified FunctionDef message, length delimited. Does not implicitly {@link tensorflow.FunctionDef.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.FunctionDef
             * @static
             * @param {tensorflow.IFunctionDef} message FunctionDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            FunctionDef.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a FunctionDef message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.FunctionDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.FunctionDef} FunctionDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a FunctionDef message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.FunctionDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.FunctionDef} FunctionDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            FunctionDef.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a FunctionDef message.
             * @function verify
             * @memberof tensorflow.FunctionDef
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a FunctionDef message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.FunctionDef
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.FunctionDef} FunctionDef
             */
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
    
            /**
             * Creates a plain object from a FunctionDef message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.FunctionDef
             * @static
             * @param {tensorflow.FunctionDef} message FunctionDef
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this FunctionDef to JSON.
             * @function toJSON
             * @memberof tensorflow.FunctionDef
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            FunctionDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return FunctionDef;
        })();
    
        tensorflow.GradientDef = (function() {
    
            /**
             * Properties of a GradientDef.
             * @memberof tensorflow
             * @interface IGradientDef
             * @property {string|null} [functionName] GradientDef functionName
             * @property {string|null} [gradientFunc] GradientDef gradientFunc
             */
    
            /**
             * Constructs a new GradientDef.
             * @memberof tensorflow
             * @classdesc Represents a GradientDef.
             * @implements IGradientDef
             * @constructor
             * @param {tensorflow.IGradientDef=} [properties] Properties to set
             */
            function GradientDef(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * GradientDef functionName.
             * @member {string} functionName
             * @memberof tensorflow.GradientDef
             * @instance
             */
            GradientDef.prototype.functionName = "";
    
            /**
             * GradientDef gradientFunc.
             * @member {string} gradientFunc
             * @memberof tensorflow.GradientDef
             * @instance
             */
            GradientDef.prototype.gradientFunc = "";
    
            /**
             * Creates a new GradientDef instance using the specified properties.
             * @function create
             * @memberof tensorflow.GradientDef
             * @static
             * @param {tensorflow.IGradientDef=} [properties] Properties to set
             * @returns {tensorflow.GradientDef} GradientDef instance
             */
            GradientDef.create = function create(properties) {
                return new GradientDef(properties);
            };
    
            /**
             * Encodes the specified GradientDef message. Does not implicitly {@link tensorflow.GradientDef.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.GradientDef
             * @static
             * @param {tensorflow.IGradientDef} message GradientDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            GradientDef.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.functionName != null && message.hasOwnProperty("functionName"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.functionName);
                if (message.gradientFunc != null && message.hasOwnProperty("gradientFunc"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.gradientFunc);
                return writer;
            };
    
            /**
             * Encodes the specified GradientDef message, length delimited. Does not implicitly {@link tensorflow.GradientDef.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.GradientDef
             * @static
             * @param {tensorflow.IGradientDef} message GradientDef message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            GradientDef.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a GradientDef message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.GradientDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.GradientDef} GradientDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a GradientDef message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.GradientDef
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.GradientDef} GradientDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            GradientDef.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a GradientDef message.
             * @function verify
             * @memberof tensorflow.GradientDef
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a GradientDef message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.GradientDef
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.GradientDef} GradientDef
             */
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
    
            /**
             * Creates a plain object from a GradientDef message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.GradientDef
             * @static
             * @param {tensorflow.GradientDef} message GradientDef
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this GradientDef to JSON.
             * @function toJSON
             * @memberof tensorflow.GradientDef
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            GradientDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return GradientDef;
        })();
    
        tensorflow.AttrValue = (function() {
    
            /**
             * Properties of an AttrValue.
             * @memberof tensorflow
             * @interface IAttrValue
             * @property {Uint8Array|null} [s] AttrValue s
             * @property {number|Long|null} [i] AttrValue i
             * @property {number|null} [f] AttrValue f
             * @property {boolean|null} [b] AttrValue b
             * @property {tensorflow.DataType|null} [type] AttrValue type
             * @property {tensorflow.ITensorShapeProto|null} [shape] AttrValue shape
             * @property {tensorflow.ITensorProto|null} [tensor] AttrValue tensor
             * @property {tensorflow.AttrValue.IListValue|null} [list] AttrValue list
             * @property {tensorflow.INameAttrList|null} [func] AttrValue func
             * @property {string|null} [placeholder] AttrValue placeholder
             */
    
            /**
             * Constructs a new AttrValue.
             * @memberof tensorflow
             * @classdesc Represents an AttrValue.
             * @implements IAttrValue
             * @constructor
             * @param {tensorflow.IAttrValue=} [properties] Properties to set
             */
            function AttrValue(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * AttrValue s.
             * @member {Uint8Array} s
             * @memberof tensorflow.AttrValue
             * @instance
             */
            AttrValue.prototype.s = $util.newBuffer([]);
    
            /**
             * AttrValue i.
             * @member {number|Long} i
             * @memberof tensorflow.AttrValue
             * @instance
             */
            AttrValue.prototype.i = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
            /**
             * AttrValue f.
             * @member {number} f
             * @memberof tensorflow.AttrValue
             * @instance
             */
            AttrValue.prototype.f = 0;
    
            /**
             * AttrValue b.
             * @member {boolean} b
             * @memberof tensorflow.AttrValue
             * @instance
             */
            AttrValue.prototype.b = false;
    
            /**
             * AttrValue type.
             * @member {tensorflow.DataType} type
             * @memberof tensorflow.AttrValue
             * @instance
             */
            AttrValue.prototype.type = 0;
    
            /**
             * AttrValue shape.
             * @member {tensorflow.ITensorShapeProto|null|undefined} shape
             * @memberof tensorflow.AttrValue
             * @instance
             */
            AttrValue.prototype.shape = null;
    
            /**
             * AttrValue tensor.
             * @member {tensorflow.ITensorProto|null|undefined} tensor
             * @memberof tensorflow.AttrValue
             * @instance
             */
            AttrValue.prototype.tensor = null;
    
            /**
             * AttrValue list.
             * @member {tensorflow.AttrValue.IListValue|null|undefined} list
             * @memberof tensorflow.AttrValue
             * @instance
             */
            AttrValue.prototype.list = null;
    
            /**
             * AttrValue func.
             * @member {tensorflow.INameAttrList|null|undefined} func
             * @memberof tensorflow.AttrValue
             * @instance
             */
            AttrValue.prototype.func = null;
    
            /**
             * AttrValue placeholder.
             * @member {string} placeholder
             * @memberof tensorflow.AttrValue
             * @instance
             */
            AttrValue.prototype.placeholder = "";
    
            // OneOf field names bound to virtual getters and setters
            var $oneOfFields;
    
            /**
             * AttrValue value.
             * @member {"s"|"i"|"f"|"b"|"type"|"shape"|"tensor"|"list"|"func"|"placeholder"|undefined} value
             * @memberof tensorflow.AttrValue
             * @instance
             */
            Object.defineProperty(AttrValue.prototype, "value", {
                get: $util.oneOfGetter($oneOfFields = ["s", "i", "f", "b", "type", "shape", "tensor", "list", "func", "placeholder"]),
                set: $util.oneOfSetter($oneOfFields)
            });
    
            /**
             * Creates a new AttrValue instance using the specified properties.
             * @function create
             * @memberof tensorflow.AttrValue
             * @static
             * @param {tensorflow.IAttrValue=} [properties] Properties to set
             * @returns {tensorflow.AttrValue} AttrValue instance
             */
            AttrValue.create = function create(properties) {
                return new AttrValue(properties);
            };
    
            /**
             * Encodes the specified AttrValue message. Does not implicitly {@link tensorflow.AttrValue.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.AttrValue
             * @static
             * @param {tensorflow.IAttrValue} message AttrValue message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AttrValue.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.list != null && message.hasOwnProperty("list"))
                    $root.tensorflow.AttrValue.ListValue.encode(message.list, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                if (message.s != null && message.hasOwnProperty("s"))
                    writer.uint32(/* id 2, wireType 2 =*/18).bytes(message.s);
                if (message.i != null && message.hasOwnProperty("i"))
                    writer.uint32(/* id 3, wireType 0 =*/24).int64(message.i);
                if (message.f != null && message.hasOwnProperty("f"))
                    writer.uint32(/* id 4, wireType 5 =*/37).float(message.f);
                if (message.b != null && message.hasOwnProperty("b"))
                    writer.uint32(/* id 5, wireType 0 =*/40).bool(message.b);
                if (message.type != null && message.hasOwnProperty("type"))
                    writer.uint32(/* id 6, wireType 0 =*/48).int32(message.type);
                if (message.shape != null && message.hasOwnProperty("shape"))
                    $root.tensorflow.TensorShapeProto.encode(message.shape, writer.uint32(/* id 7, wireType 2 =*/58).fork()).ldelim();
                if (message.tensor != null && message.hasOwnProperty("tensor"))
                    $root.tensorflow.TensorProto.encode(message.tensor, writer.uint32(/* id 8, wireType 2 =*/66).fork()).ldelim();
                if (message.placeholder != null && message.hasOwnProperty("placeholder"))
                    writer.uint32(/* id 9, wireType 2 =*/74).string(message.placeholder);
                if (message.func != null && message.hasOwnProperty("func"))
                    $root.tensorflow.NameAttrList.encode(message.func, writer.uint32(/* id 10, wireType 2 =*/82).fork()).ldelim();
                return writer;
            };
    
            /**
             * Encodes the specified AttrValue message, length delimited. Does not implicitly {@link tensorflow.AttrValue.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.AttrValue
             * @static
             * @param {tensorflow.IAttrValue} message AttrValue message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AttrValue.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes an AttrValue message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.AttrValue
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.AttrValue} AttrValue
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes an AttrValue message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.AttrValue
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.AttrValue} AttrValue
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AttrValue.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies an AttrValue message.
             * @function verify
             * @memberof tensorflow.AttrValue
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates an AttrValue message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.AttrValue
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.AttrValue} AttrValue
             */
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
    
            /**
             * Creates a plain object from an AttrValue message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.AttrValue
             * @static
             * @param {tensorflow.AttrValue} message AttrValue
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this AttrValue to JSON.
             * @function toJSON
             * @memberof tensorflow.AttrValue
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            AttrValue.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            AttrValue.ListValue = (function() {
    
                /**
                 * Properties of a ListValue.
                 * @memberof tensorflow.AttrValue
                 * @interface IListValue
                 * @property {Array.<Uint8Array>|null} [s] ListValue s
                 * @property {Array.<number|Long>|null} [i] ListValue i
                 * @property {Array.<number>|null} [f] ListValue f
                 * @property {Array.<boolean>|null} [b] ListValue b
                 * @property {Array.<tensorflow.DataType>|null} [type] ListValue type
                 * @property {Array.<tensorflow.ITensorShapeProto>|null} [shape] ListValue shape
                 * @property {Array.<tensorflow.ITensorProto>|null} [tensor] ListValue tensor
                 * @property {Array.<tensorflow.INameAttrList>|null} [func] ListValue func
                 */
    
                /**
                 * Constructs a new ListValue.
                 * @memberof tensorflow.AttrValue
                 * @classdesc Represents a ListValue.
                 * @implements IListValue
                 * @constructor
                 * @param {tensorflow.AttrValue.IListValue=} [properties] Properties to set
                 */
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
    
                /**
                 * ListValue s.
                 * @member {Array.<Uint8Array>} s
                 * @memberof tensorflow.AttrValue.ListValue
                 * @instance
                 */
                ListValue.prototype.s = $util.emptyArray;
    
                /**
                 * ListValue i.
                 * @member {Array.<number|Long>} i
                 * @memberof tensorflow.AttrValue.ListValue
                 * @instance
                 */
                ListValue.prototype.i = $util.emptyArray;
    
                /**
                 * ListValue f.
                 * @member {Array.<number>} f
                 * @memberof tensorflow.AttrValue.ListValue
                 * @instance
                 */
                ListValue.prototype.f = $util.emptyArray;
    
                /**
                 * ListValue b.
                 * @member {Array.<boolean>} b
                 * @memberof tensorflow.AttrValue.ListValue
                 * @instance
                 */
                ListValue.prototype.b = $util.emptyArray;
    
                /**
                 * ListValue type.
                 * @member {Array.<tensorflow.DataType>} type
                 * @memberof tensorflow.AttrValue.ListValue
                 * @instance
                 */
                ListValue.prototype.type = $util.emptyArray;
    
                /**
                 * ListValue shape.
                 * @member {Array.<tensorflow.ITensorShapeProto>} shape
                 * @memberof tensorflow.AttrValue.ListValue
                 * @instance
                 */
                ListValue.prototype.shape = $util.emptyArray;
    
                /**
                 * ListValue tensor.
                 * @member {Array.<tensorflow.ITensorProto>} tensor
                 * @memberof tensorflow.AttrValue.ListValue
                 * @instance
                 */
                ListValue.prototype.tensor = $util.emptyArray;
    
                /**
                 * ListValue func.
                 * @member {Array.<tensorflow.INameAttrList>} func
                 * @memberof tensorflow.AttrValue.ListValue
                 * @instance
                 */
                ListValue.prototype.func = $util.emptyArray;
    
                /**
                 * Creates a new ListValue instance using the specified properties.
                 * @function create
                 * @memberof tensorflow.AttrValue.ListValue
                 * @static
                 * @param {tensorflow.AttrValue.IListValue=} [properties] Properties to set
                 * @returns {tensorflow.AttrValue.ListValue} ListValue instance
                 */
                ListValue.create = function create(properties) {
                    return new ListValue(properties);
                };
    
                /**
                 * Encodes the specified ListValue message. Does not implicitly {@link tensorflow.AttrValue.ListValue.verify|verify} messages.
                 * @function encode
                 * @memberof tensorflow.AttrValue.ListValue
                 * @static
                 * @param {tensorflow.AttrValue.IListValue} message ListValue message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                ListValue.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.s != null && message.s.length)
                        for (var i = 0; i < message.s.length; ++i)
                            writer.uint32(/* id 2, wireType 2 =*/18).bytes(message.s[i]);
                    if (message.i != null && message.i.length) {
                        writer.uint32(/* id 3, wireType 2 =*/26).fork();
                        for (var i = 0; i < message.i.length; ++i)
                            writer.int64(message.i[i]);
                        writer.ldelim();
                    }
                    if (message.f != null && message.f.length) {
                        writer.uint32(/* id 4, wireType 2 =*/34).fork();
                        for (var i = 0; i < message.f.length; ++i)
                            writer.float(message.f[i]);
                        writer.ldelim();
                    }
                    if (message.b != null && message.b.length) {
                        writer.uint32(/* id 5, wireType 2 =*/42).fork();
                        for (var i = 0; i < message.b.length; ++i)
                            writer.bool(message.b[i]);
                        writer.ldelim();
                    }
                    if (message.type != null && message.type.length) {
                        writer.uint32(/* id 6, wireType 2 =*/50).fork();
                        for (var i = 0; i < message.type.length; ++i)
                            writer.int32(message.type[i]);
                        writer.ldelim();
                    }
                    if (message.shape != null && message.shape.length)
                        for (var i = 0; i < message.shape.length; ++i)
                            $root.tensorflow.TensorShapeProto.encode(message.shape[i], writer.uint32(/* id 7, wireType 2 =*/58).fork()).ldelim();
                    if (message.tensor != null && message.tensor.length)
                        for (var i = 0; i < message.tensor.length; ++i)
                            $root.tensorflow.TensorProto.encode(message.tensor[i], writer.uint32(/* id 8, wireType 2 =*/66).fork()).ldelim();
                    if (message.func != null && message.func.length)
                        for (var i = 0; i < message.func.length; ++i)
                            $root.tensorflow.NameAttrList.encode(message.func[i], writer.uint32(/* id 9, wireType 2 =*/74).fork()).ldelim();
                    return writer;
                };
    
                /**
                 * Encodes the specified ListValue message, length delimited. Does not implicitly {@link tensorflow.AttrValue.ListValue.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof tensorflow.AttrValue.ListValue
                 * @static
                 * @param {tensorflow.AttrValue.IListValue} message ListValue message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                ListValue.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes a ListValue message from the specified reader or buffer.
                 * @function decode
                 * @memberof tensorflow.AttrValue.ListValue
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {tensorflow.AttrValue.ListValue} ListValue
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
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
    
                /**
                 * Decodes a ListValue message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof tensorflow.AttrValue.ListValue
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {tensorflow.AttrValue.ListValue} ListValue
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                ListValue.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies a ListValue message.
                 * @function verify
                 * @memberof tensorflow.AttrValue.ListValue
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
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
    
                /**
                 * Creates a ListValue message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof tensorflow.AttrValue.ListValue
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {tensorflow.AttrValue.ListValue} ListValue
                 */
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
    
                /**
                 * Creates a plain object from a ListValue message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof tensorflow.AttrValue.ListValue
                 * @static
                 * @param {tensorflow.AttrValue.ListValue} message ListValue
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
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
    
                /**
                 * Converts this ListValue to JSON.
                 * @function toJSON
                 * @memberof tensorflow.AttrValue.ListValue
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                ListValue.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return ListValue;
            })();
    
            return AttrValue;
        })();
    
        tensorflow.NameAttrList = (function() {
    
            /**
             * Properties of a NameAttrList.
             * @memberof tensorflow
             * @interface INameAttrList
             * @property {string|null} [name] NameAttrList name
             * @property {Object.<string,tensorflow.IAttrValue>|null} [attr] NameAttrList attr
             */
    
            /**
             * Constructs a new NameAttrList.
             * @memberof tensorflow
             * @classdesc Represents a NameAttrList.
             * @implements INameAttrList
             * @constructor
             * @param {tensorflow.INameAttrList=} [properties] Properties to set
             */
            function NameAttrList(properties) {
                this.attr = {};
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * NameAttrList name.
             * @member {string} name
             * @memberof tensorflow.NameAttrList
             * @instance
             */
            NameAttrList.prototype.name = "";
    
            /**
             * NameAttrList attr.
             * @member {Object.<string,tensorflow.IAttrValue>} attr
             * @memberof tensorflow.NameAttrList
             * @instance
             */
            NameAttrList.prototype.attr = $util.emptyObject;
    
            /**
             * Creates a new NameAttrList instance using the specified properties.
             * @function create
             * @memberof tensorflow.NameAttrList
             * @static
             * @param {tensorflow.INameAttrList=} [properties] Properties to set
             * @returns {tensorflow.NameAttrList} NameAttrList instance
             */
            NameAttrList.create = function create(properties) {
                return new NameAttrList(properties);
            };
    
            /**
             * Encodes the specified NameAttrList message. Does not implicitly {@link tensorflow.NameAttrList.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.NameAttrList
             * @static
             * @param {tensorflow.INameAttrList} message NameAttrList message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            NameAttrList.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.name != null && message.hasOwnProperty("name"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
                if (message.attr != null && message.hasOwnProperty("attr"))
                    for (var keys = Object.keys(message.attr), i = 0; i < keys.length; ++i) {
                        writer.uint32(/* id 2, wireType 2 =*/18).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]);
                        $root.tensorflow.AttrValue.encode(message.attr[keys[i]], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim().ldelim();
                    }
                return writer;
            };
    
            /**
             * Encodes the specified NameAttrList message, length delimited. Does not implicitly {@link tensorflow.NameAttrList.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.NameAttrList
             * @static
             * @param {tensorflow.INameAttrList} message NameAttrList message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            NameAttrList.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a NameAttrList message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.NameAttrList
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.NameAttrList} NameAttrList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a NameAttrList message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.NameAttrList
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.NameAttrList} NameAttrList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            NameAttrList.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a NameAttrList message.
             * @function verify
             * @memberof tensorflow.NameAttrList
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a NameAttrList message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.NameAttrList
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.NameAttrList} NameAttrList
             */
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
    
            /**
             * Creates a plain object from a NameAttrList message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.NameAttrList
             * @static
             * @param {tensorflow.NameAttrList} message NameAttrList
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this NameAttrList to JSON.
             * @function toJSON
             * @memberof tensorflow.NameAttrList
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            NameAttrList.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return NameAttrList;
        })();
    
        tensorflow.TensorProto = (function() {
    
            /**
             * Properties of a TensorProto.
             * @memberof tensorflow
             * @interface ITensorProto
             * @property {tensorflow.DataType|null} [dtype] TensorProto dtype
             * @property {tensorflow.ITensorShapeProto|null} [tensorShape] TensorProto tensorShape
             * @property {number|null} [versionNumber] TensorProto versionNumber
             * @property {Uint8Array|null} [tensorContent] TensorProto tensorContent
             * @property {Array.<number>|null} [halfVal] TensorProto halfVal
             * @property {Array.<number>|null} [floatVal] TensorProto floatVal
             * @property {Array.<number>|null} [doubleVal] TensorProto doubleVal
             * @property {Array.<number>|null} [intVal] TensorProto intVal
             * @property {Array.<Uint8Array>|null} [stringVal] TensorProto stringVal
             * @property {Array.<number>|null} [scomplexVal] TensorProto scomplexVal
             * @property {Array.<number|Long>|null} [int64Val] TensorProto int64Val
             * @property {Array.<boolean>|null} [boolVal] TensorProto boolVal
             * @property {Array.<number>|null} [dcomplexVal] TensorProto dcomplexVal
             * @property {Array.<tensorflow.IResourceHandleProto>|null} [resourceHandleVal] TensorProto resourceHandleVal
             * @property {Array.<tensorflow.IVariantTensorDataProto>|null} [variantVal] TensorProto variantVal
             * @property {Array.<number>|null} [uint32Val] TensorProto uint32Val
             * @property {Array.<number|Long>|null} [uint64Val] TensorProto uint64Val
             */
    
            /**
             * Constructs a new TensorProto.
             * @memberof tensorflow
             * @classdesc Represents a TensorProto.
             * @implements ITensorProto
             * @constructor
             * @param {tensorflow.ITensorProto=} [properties] Properties to set
             */
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
    
            /**
             * TensorProto dtype.
             * @member {tensorflow.DataType} dtype
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.dtype = 0;
    
            /**
             * TensorProto tensorShape.
             * @member {tensorflow.ITensorShapeProto|null|undefined} tensorShape
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.tensorShape = null;
    
            /**
             * TensorProto versionNumber.
             * @member {number} versionNumber
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.versionNumber = 0;
    
            /**
             * TensorProto tensorContent.
             * @member {Uint8Array} tensorContent
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.tensorContent = $util.newBuffer([]);
    
            /**
             * TensorProto halfVal.
             * @member {Array.<number>} halfVal
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.halfVal = $util.emptyArray;
    
            /**
             * TensorProto floatVal.
             * @member {Array.<number>} floatVal
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.floatVal = $util.emptyArray;
    
            /**
             * TensorProto doubleVal.
             * @member {Array.<number>} doubleVal
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.doubleVal = $util.emptyArray;
    
            /**
             * TensorProto intVal.
             * @member {Array.<number>} intVal
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.intVal = $util.emptyArray;
    
            /**
             * TensorProto stringVal.
             * @member {Array.<Uint8Array>} stringVal
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.stringVal = $util.emptyArray;
    
            /**
             * TensorProto scomplexVal.
             * @member {Array.<number>} scomplexVal
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.scomplexVal = $util.emptyArray;
    
            /**
             * TensorProto int64Val.
             * @member {Array.<number|Long>} int64Val
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.int64Val = $util.emptyArray;
    
            /**
             * TensorProto boolVal.
             * @member {Array.<boolean>} boolVal
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.boolVal = $util.emptyArray;
    
            /**
             * TensorProto dcomplexVal.
             * @member {Array.<number>} dcomplexVal
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.dcomplexVal = $util.emptyArray;
    
            /**
             * TensorProto resourceHandleVal.
             * @member {Array.<tensorflow.IResourceHandleProto>} resourceHandleVal
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.resourceHandleVal = $util.emptyArray;
    
            /**
             * TensorProto variantVal.
             * @member {Array.<tensorflow.IVariantTensorDataProto>} variantVal
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.variantVal = $util.emptyArray;
    
            /**
             * TensorProto uint32Val.
             * @member {Array.<number>} uint32Val
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.uint32Val = $util.emptyArray;
    
            /**
             * TensorProto uint64Val.
             * @member {Array.<number|Long>} uint64Val
             * @memberof tensorflow.TensorProto
             * @instance
             */
            TensorProto.prototype.uint64Val = $util.emptyArray;
    
            /**
             * Creates a new TensorProto instance using the specified properties.
             * @function create
             * @memberof tensorflow.TensorProto
             * @static
             * @param {tensorflow.ITensorProto=} [properties] Properties to set
             * @returns {tensorflow.TensorProto} TensorProto instance
             */
            TensorProto.create = function create(properties) {
                return new TensorProto(properties);
            };
    
            /**
             * Encodes the specified TensorProto message. Does not implicitly {@link tensorflow.TensorProto.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.TensorProto
             * @static
             * @param {tensorflow.ITensorProto} message TensorProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TensorProto.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.dtype != null && message.hasOwnProperty("dtype"))
                    writer.uint32(/* id 1, wireType 0 =*/8).int32(message.dtype);
                if (message.tensorShape != null && message.hasOwnProperty("tensorShape"))
                    $root.tensorflow.TensorShapeProto.encode(message.tensorShape, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                if (message.versionNumber != null && message.hasOwnProperty("versionNumber"))
                    writer.uint32(/* id 3, wireType 0 =*/24).int32(message.versionNumber);
                if (message.tensorContent != null && message.hasOwnProperty("tensorContent"))
                    writer.uint32(/* id 4, wireType 2 =*/34).bytes(message.tensorContent);
                if (message.floatVal != null && message.floatVal.length) {
                    writer.uint32(/* id 5, wireType 2 =*/42).fork();
                    for (var i = 0; i < message.floatVal.length; ++i)
                        writer.float(message.floatVal[i]);
                    writer.ldelim();
                }
                if (message.doubleVal != null && message.doubleVal.length) {
                    writer.uint32(/* id 6, wireType 2 =*/50).fork();
                    for (var i = 0; i < message.doubleVal.length; ++i)
                        writer.double(message.doubleVal[i]);
                    writer.ldelim();
                }
                if (message.intVal != null && message.intVal.length) {
                    writer.uint32(/* id 7, wireType 2 =*/58).fork();
                    for (var i = 0; i < message.intVal.length; ++i)
                        writer.int32(message.intVal[i]);
                    writer.ldelim();
                }
                if (message.stringVal != null && message.stringVal.length)
                    for (var i = 0; i < message.stringVal.length; ++i)
                        writer.uint32(/* id 8, wireType 2 =*/66).bytes(message.stringVal[i]);
                if (message.scomplexVal != null && message.scomplexVal.length) {
                    writer.uint32(/* id 9, wireType 2 =*/74).fork();
                    for (var i = 0; i < message.scomplexVal.length; ++i)
                        writer.float(message.scomplexVal[i]);
                    writer.ldelim();
                }
                if (message.int64Val != null && message.int64Val.length) {
                    writer.uint32(/* id 10, wireType 2 =*/82).fork();
                    for (var i = 0; i < message.int64Val.length; ++i)
                        writer.int64(message.int64Val[i]);
                    writer.ldelim();
                }
                if (message.boolVal != null && message.boolVal.length) {
                    writer.uint32(/* id 11, wireType 2 =*/90).fork();
                    for (var i = 0; i < message.boolVal.length; ++i)
                        writer.bool(message.boolVal[i]);
                    writer.ldelim();
                }
                if (message.dcomplexVal != null && message.dcomplexVal.length) {
                    writer.uint32(/* id 12, wireType 2 =*/98).fork();
                    for (var i = 0; i < message.dcomplexVal.length; ++i)
                        writer.double(message.dcomplexVal[i]);
                    writer.ldelim();
                }
                if (message.halfVal != null && message.halfVal.length) {
                    writer.uint32(/* id 13, wireType 2 =*/106).fork();
                    for (var i = 0; i < message.halfVal.length; ++i)
                        writer.int32(message.halfVal[i]);
                    writer.ldelim();
                }
                if (message.resourceHandleVal != null && message.resourceHandleVal.length)
                    for (var i = 0; i < message.resourceHandleVal.length; ++i)
                        $root.tensorflow.ResourceHandleProto.encode(message.resourceHandleVal[i], writer.uint32(/* id 14, wireType 2 =*/114).fork()).ldelim();
                if (message.variantVal != null && message.variantVal.length)
                    for (var i = 0; i < message.variantVal.length; ++i)
                        $root.tensorflow.VariantTensorDataProto.encode(message.variantVal[i], writer.uint32(/* id 15, wireType 2 =*/122).fork()).ldelim();
                if (message.uint32Val != null && message.uint32Val.length) {
                    writer.uint32(/* id 16, wireType 2 =*/130).fork();
                    for (var i = 0; i < message.uint32Val.length; ++i)
                        writer.uint32(message.uint32Val[i]);
                    writer.ldelim();
                }
                if (message.uint64Val != null && message.uint64Val.length) {
                    writer.uint32(/* id 17, wireType 2 =*/138).fork();
                    for (var i = 0; i < message.uint64Val.length; ++i)
                        writer.uint64(message.uint64Val[i]);
                    writer.ldelim();
                }
                return writer;
            };
    
            /**
             * Encodes the specified TensorProto message, length delimited. Does not implicitly {@link tensorflow.TensorProto.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.TensorProto
             * @static
             * @param {tensorflow.ITensorProto} message TensorProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TensorProto.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a TensorProto message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.TensorProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.TensorProto} TensorProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a TensorProto message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.TensorProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.TensorProto} TensorProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            TensorProto.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a TensorProto message.
             * @function verify
             * @memberof tensorflow.TensorProto
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a TensorProto message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.TensorProto
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.TensorProto} TensorProto
             */
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
    
            /**
             * Creates a plain object from a TensorProto message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.TensorProto
             * @static
             * @param {tensorflow.TensorProto} message TensorProto
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this TensorProto to JSON.
             * @function toJSON
             * @memberof tensorflow.TensorProto
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            TensorProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return TensorProto;
        })();
    
        tensorflow.VariantTensorDataProto = (function() {
    
            /**
             * Properties of a VariantTensorDataProto.
             * @memberof tensorflow
             * @interface IVariantTensorDataProto
             * @property {string|null} [typeName] VariantTensorDataProto typeName
             * @property {Uint8Array|null} [metadata] VariantTensorDataProto metadata
             * @property {Array.<tensorflow.ITensorProto>|null} [tensors] VariantTensorDataProto tensors
             */
    
            /**
             * Constructs a new VariantTensorDataProto.
             * @memberof tensorflow
             * @classdesc Represents a VariantTensorDataProto.
             * @implements IVariantTensorDataProto
             * @constructor
             * @param {tensorflow.IVariantTensorDataProto=} [properties] Properties to set
             */
            function VariantTensorDataProto(properties) {
                this.tensors = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * VariantTensorDataProto typeName.
             * @member {string} typeName
             * @memberof tensorflow.VariantTensorDataProto
             * @instance
             */
            VariantTensorDataProto.prototype.typeName = "";
    
            /**
             * VariantTensorDataProto metadata.
             * @member {Uint8Array} metadata
             * @memberof tensorflow.VariantTensorDataProto
             * @instance
             */
            VariantTensorDataProto.prototype.metadata = $util.newBuffer([]);
    
            /**
             * VariantTensorDataProto tensors.
             * @member {Array.<tensorflow.ITensorProto>} tensors
             * @memberof tensorflow.VariantTensorDataProto
             * @instance
             */
            VariantTensorDataProto.prototype.tensors = $util.emptyArray;
    
            /**
             * Creates a new VariantTensorDataProto instance using the specified properties.
             * @function create
             * @memberof tensorflow.VariantTensorDataProto
             * @static
             * @param {tensorflow.IVariantTensorDataProto=} [properties] Properties to set
             * @returns {tensorflow.VariantTensorDataProto} VariantTensorDataProto instance
             */
            VariantTensorDataProto.create = function create(properties) {
                return new VariantTensorDataProto(properties);
            };
    
            /**
             * Encodes the specified VariantTensorDataProto message. Does not implicitly {@link tensorflow.VariantTensorDataProto.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.VariantTensorDataProto
             * @static
             * @param {tensorflow.IVariantTensorDataProto} message VariantTensorDataProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            VariantTensorDataProto.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.typeName != null && message.hasOwnProperty("typeName"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.typeName);
                if (message.metadata != null && message.hasOwnProperty("metadata"))
                    writer.uint32(/* id 2, wireType 2 =*/18).bytes(message.metadata);
                if (message.tensors != null && message.tensors.length)
                    for (var i = 0; i < message.tensors.length; ++i)
                        $root.tensorflow.TensorProto.encode(message.tensors[i], writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
                return writer;
            };
    
            /**
             * Encodes the specified VariantTensorDataProto message, length delimited. Does not implicitly {@link tensorflow.VariantTensorDataProto.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.VariantTensorDataProto
             * @static
             * @param {tensorflow.IVariantTensorDataProto} message VariantTensorDataProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            VariantTensorDataProto.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a VariantTensorDataProto message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.VariantTensorDataProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.VariantTensorDataProto} VariantTensorDataProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a VariantTensorDataProto message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.VariantTensorDataProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.VariantTensorDataProto} VariantTensorDataProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            VariantTensorDataProto.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a VariantTensorDataProto message.
             * @function verify
             * @memberof tensorflow.VariantTensorDataProto
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a VariantTensorDataProto message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.VariantTensorDataProto
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.VariantTensorDataProto} VariantTensorDataProto
             */
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
    
            /**
             * Creates a plain object from a VariantTensorDataProto message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.VariantTensorDataProto
             * @static
             * @param {tensorflow.VariantTensorDataProto} message VariantTensorDataProto
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this VariantTensorDataProto to JSON.
             * @function toJSON
             * @memberof tensorflow.VariantTensorDataProto
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            VariantTensorDataProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return VariantTensorDataProto;
        })();
    
        tensorflow.ResourceHandleProto = (function() {
    
            /**
             * Properties of a ResourceHandleProto.
             * @memberof tensorflow
             * @interface IResourceHandleProto
             * @property {string|null} [device] ResourceHandleProto device
             * @property {string|null} [container] ResourceHandleProto container
             * @property {string|null} [name] ResourceHandleProto name
             * @property {number|Long|null} [hashCode] ResourceHandleProto hashCode
             * @property {string|null} [maybeTypeName] ResourceHandleProto maybeTypeName
             */
    
            /**
             * Constructs a new ResourceHandleProto.
             * @memberof tensorflow
             * @classdesc Represents a ResourceHandleProto.
             * @implements IResourceHandleProto
             * @constructor
             * @param {tensorflow.IResourceHandleProto=} [properties] Properties to set
             */
            function ResourceHandleProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * ResourceHandleProto device.
             * @member {string} device
             * @memberof tensorflow.ResourceHandleProto
             * @instance
             */
            ResourceHandleProto.prototype.device = "";
    
            /**
             * ResourceHandleProto container.
             * @member {string} container
             * @memberof tensorflow.ResourceHandleProto
             * @instance
             */
            ResourceHandleProto.prototype.container = "";
    
            /**
             * ResourceHandleProto name.
             * @member {string} name
             * @memberof tensorflow.ResourceHandleProto
             * @instance
             */
            ResourceHandleProto.prototype.name = "";
    
            /**
             * ResourceHandleProto hashCode.
             * @member {number|Long} hashCode
             * @memberof tensorflow.ResourceHandleProto
             * @instance
             */
            ResourceHandleProto.prototype.hashCode = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
    
            /**
             * ResourceHandleProto maybeTypeName.
             * @member {string} maybeTypeName
             * @memberof tensorflow.ResourceHandleProto
             * @instance
             */
            ResourceHandleProto.prototype.maybeTypeName = "";
    
            /**
             * Creates a new ResourceHandleProto instance using the specified properties.
             * @function create
             * @memberof tensorflow.ResourceHandleProto
             * @static
             * @param {tensorflow.IResourceHandleProto=} [properties] Properties to set
             * @returns {tensorflow.ResourceHandleProto} ResourceHandleProto instance
             */
            ResourceHandleProto.create = function create(properties) {
                return new ResourceHandleProto(properties);
            };
    
            /**
             * Encodes the specified ResourceHandleProto message. Does not implicitly {@link tensorflow.ResourceHandleProto.verify|verify} messages.
             * @function encode
             * @memberof tensorflow.ResourceHandleProto
             * @static
             * @param {tensorflow.IResourceHandleProto} message ResourceHandleProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ResourceHandleProto.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.device != null && message.hasOwnProperty("device"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.device);
                if (message.container != null && message.hasOwnProperty("container"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.container);
                if (message.name != null && message.hasOwnProperty("name"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.name);
                if (message.hashCode != null && message.hasOwnProperty("hashCode"))
                    writer.uint32(/* id 4, wireType 0 =*/32).uint64(message.hashCode);
                if (message.maybeTypeName != null && message.hasOwnProperty("maybeTypeName"))
                    writer.uint32(/* id 5, wireType 2 =*/42).string(message.maybeTypeName);
                return writer;
            };
    
            /**
             * Encodes the specified ResourceHandleProto message, length delimited. Does not implicitly {@link tensorflow.ResourceHandleProto.verify|verify} messages.
             * @function encodeDelimited
             * @memberof tensorflow.ResourceHandleProto
             * @static
             * @param {tensorflow.IResourceHandleProto} message ResourceHandleProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ResourceHandleProto.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a ResourceHandleProto message from the specified reader or buffer.
             * @function decode
             * @memberof tensorflow.ResourceHandleProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {tensorflow.ResourceHandleProto} ResourceHandleProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a ResourceHandleProto message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof tensorflow.ResourceHandleProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {tensorflow.ResourceHandleProto} ResourceHandleProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ResourceHandleProto.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a ResourceHandleProto message.
             * @function verify
             * @memberof tensorflow.ResourceHandleProto
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
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
    
            /**
             * Creates a ResourceHandleProto message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof tensorflow.ResourceHandleProto
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {tensorflow.ResourceHandleProto} ResourceHandleProto
             */
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
    
            /**
             * Creates a plain object from a ResourceHandleProto message. Also converts values to other types if specified.
             * @function toObject
             * @memberof tensorflow.ResourceHandleProto
             * @static
             * @param {tensorflow.ResourceHandleProto} message ResourceHandleProto
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
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
    
            /**
             * Converts this ResourceHandleProto to JSON.
             * @function toJSON
             * @memberof tensorflow.ResourceHandleProto
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            ResourceHandleProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ResourceHandleProto;
        })();
    
        return tensorflow;
    })();
    
    $root.google = (function() {
    
        /**
         * Namespace google.
         * @exports google
         * @namespace
         */
        var google = {};
    
        google.protobuf = (function() {
    
            /**
             * Namespace protobuf.
             * @memberof google
             * @namespace
             */
            var protobuf = {};
    
            protobuf.Any = (function() {
    
                /**
                 * Properties of an Any.
                 * @memberof google.protobuf
                 * @interface IAny
                 * @property {string|null} [type_url] Any type_url
                 * @property {Uint8Array|null} [value] Any value
                 */
    
                /**
                 * Constructs a new Any.
                 * @memberof google.protobuf
                 * @classdesc Represents an Any.
                 * @implements IAny
                 * @constructor
                 * @param {google.protobuf.IAny=} [properties] Properties to set
                 */
                function Any(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * Any type_url.
                 * @member {string} type_url
                 * @memberof google.protobuf.Any
                 * @instance
                 */
                Any.prototype.type_url = "";
    
                /**
                 * Any value.
                 * @member {Uint8Array} value
                 * @memberof google.protobuf.Any
                 * @instance
                 */
                Any.prototype.value = $util.newBuffer([]);
    
                /**
                 * Creates a new Any instance using the specified properties.
                 * @function create
                 * @memberof google.protobuf.Any
                 * @static
                 * @param {google.protobuf.IAny=} [properties] Properties to set
                 * @returns {google.protobuf.Any} Any instance
                 */
                Any.create = function create(properties) {
                    return new Any(properties);
                };
    
                /**
                 * Encodes the specified Any message. Does not implicitly {@link google.protobuf.Any.verify|verify} messages.
                 * @function encode
                 * @memberof google.protobuf.Any
                 * @static
                 * @param {google.protobuf.IAny} message Any message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Any.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.type_url != null && message.hasOwnProperty("type_url"))
                        writer.uint32(/* id 1, wireType 2 =*/10).string(message.type_url);
                    if (message.value != null && message.hasOwnProperty("value"))
                        writer.uint32(/* id 2, wireType 2 =*/18).bytes(message.value);
                    return writer;
                };
    
                /**
                 * Encodes the specified Any message, length delimited. Does not implicitly {@link google.protobuf.Any.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof google.protobuf.Any
                 * @static
                 * @param {google.protobuf.IAny} message Any message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Any.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes an Any message from the specified reader or buffer.
                 * @function decode
                 * @memberof google.protobuf.Any
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {google.protobuf.Any} Any
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
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
    
                /**
                 * Decodes an Any message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof google.protobuf.Any
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {google.protobuf.Any} Any
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                Any.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies an Any message.
                 * @function verify
                 * @memberof google.protobuf.Any
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
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
    
                /**
                 * Creates an Any message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof google.protobuf.Any
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {google.protobuf.Any} Any
                 */
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
    
                /**
                 * Creates a plain object from an Any message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof google.protobuf.Any
                 * @static
                 * @param {google.protobuf.Any} message Any
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
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
    
                /**
                 * Converts this Any to JSON.
                 * @function toJSON
                 * @memberof google.protobuf.Any
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
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
