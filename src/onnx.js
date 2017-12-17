/*eslint-disable block-scoped-var, no-redeclare, no-control-regex, no-prototype-builtins*/
(function($protobuf) {
    "use strict";

    // Common aliases
    var $Reader = $protobuf.Reader, $Writer = $protobuf.Writer, $util = $protobuf.util;
    
    // Exported root namespace
    var $root = $protobuf.roots.onnx || ($protobuf.roots.onnx = {});
    
    $root.onnx = (function() {
    
        /**
         * Namespace onnx.
         * @exports onnx
         * @namespace
         */
        var onnx = {};
    
        /**
         * Version enum.
         * @name onnx.Version
         * @enum {string}
         * @property {number} _START_VERSION=0 _START_VERSION value
         * @property {number} IR_VERSION_2017_10_10=1 IR_VERSION_2017_10_10 value
         * @property {number} IR_VERSION_2017_10_30=2 IR_VERSION_2017_10_30 value
         * @property {number} IR_VERSION=3 IR_VERSION value
         */
        onnx.Version = (function() {
            var valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "_START_VERSION"] = 0;
            values[valuesById[1] = "IR_VERSION_2017_10_10"] = 1;
            values[valuesById[2] = "IR_VERSION_2017_10_30"] = 2;
            values[valuesById[3] = "IR_VERSION"] = 3;
            return values;
        })();
    
        onnx.AttributeProto = (function() {
    
            /**
             * Properties of an AttributeProto.
             * @memberof onnx
             * @interface IAttributeProto
             * @property {string|null} [name] AttributeProto name
             * @property {string|null} [docString] AttributeProto docString
             * @property {onnx.AttributeProto.AttributeType|null} [type] AttributeProto type
             * @property {number|null} [f] AttributeProto f
             * @property {number|Long|null} [i] AttributeProto i
             * @property {Uint8Array|null} [s] AttributeProto s
             * @property {onnx.ITensorProto|null} [t] AttributeProto t
             * @property {onnx.IGraphProto|null} [g] AttributeProto g
             * @property {Array.<number>|null} [floats] AttributeProto floats
             * @property {Array.<number|Long>|null} [ints] AttributeProto ints
             * @property {Array.<Uint8Array>|null} [strings] AttributeProto strings
             * @property {Array.<onnx.ITensorProto>|null} [tensors] AttributeProto tensors
             * @property {Array.<onnx.IGraphProto>|null} [graphs] AttributeProto graphs
             */
    
            /**
             * Constructs a new AttributeProto.
             * @memberof onnx
             * @classdesc Represents an AttributeProto.
             * @implements IAttributeProto
             * @constructor
             * @param {onnx.IAttributeProto=} [properties] Properties to set
             */
            function AttributeProto(properties) {
                this.floats = [];
                this.ints = [];
                this.strings = [];
                this.tensors = [];
                this.graphs = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * AttributeProto name.
             * @member {string} name
             * @memberof onnx.AttributeProto
             * @instance
             */
            AttributeProto.prototype.name = "";
    
            /**
             * AttributeProto docString.
             * @member {string} docString
             * @memberof onnx.AttributeProto
             * @instance
             */
            AttributeProto.prototype.docString = "";
    
            /**
             * AttributeProto type.
             * @member {onnx.AttributeProto.AttributeType} type
             * @memberof onnx.AttributeProto
             * @instance
             */
            AttributeProto.prototype.type = 0;
    
            /**
             * AttributeProto f.
             * @member {number} f
             * @memberof onnx.AttributeProto
             * @instance
             */
            AttributeProto.prototype.f = 0;
    
            /**
             * AttributeProto i.
             * @member {number|Long} i
             * @memberof onnx.AttributeProto
             * @instance
             */
            AttributeProto.prototype.i = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
            /**
             * AttributeProto s.
             * @member {Uint8Array} s
             * @memberof onnx.AttributeProto
             * @instance
             */
            AttributeProto.prototype.s = $util.newBuffer([]);
    
            /**
             * AttributeProto t.
             * @member {onnx.ITensorProto|null|undefined} t
             * @memberof onnx.AttributeProto
             * @instance
             */
            AttributeProto.prototype.t = null;
    
            /**
             * AttributeProto g.
             * @member {onnx.IGraphProto|null|undefined} g
             * @memberof onnx.AttributeProto
             * @instance
             */
            AttributeProto.prototype.g = null;
    
            /**
             * AttributeProto floats.
             * @member {Array.<number>} floats
             * @memberof onnx.AttributeProto
             * @instance
             */
            AttributeProto.prototype.floats = $util.emptyArray;
    
            /**
             * AttributeProto ints.
             * @member {Array.<number|Long>} ints
             * @memberof onnx.AttributeProto
             * @instance
             */
            AttributeProto.prototype.ints = $util.emptyArray;
    
            /**
             * AttributeProto strings.
             * @member {Array.<Uint8Array>} strings
             * @memberof onnx.AttributeProto
             * @instance
             */
            AttributeProto.prototype.strings = $util.emptyArray;
    
            /**
             * AttributeProto tensors.
             * @member {Array.<onnx.ITensorProto>} tensors
             * @memberof onnx.AttributeProto
             * @instance
             */
            AttributeProto.prototype.tensors = $util.emptyArray;
    
            /**
             * AttributeProto graphs.
             * @member {Array.<onnx.IGraphProto>} graphs
             * @memberof onnx.AttributeProto
             * @instance
             */
            AttributeProto.prototype.graphs = $util.emptyArray;
    
            /**
             * Creates a new AttributeProto instance using the specified properties.
             * @function create
             * @memberof onnx.AttributeProto
             * @static
             * @param {onnx.IAttributeProto=} [properties] Properties to set
             * @returns {onnx.AttributeProto} AttributeProto instance
             */
            AttributeProto.create = function create(properties) {
                return new AttributeProto(properties);
            };
    
            /**
             * Encodes the specified AttributeProto message. Does not implicitly {@link onnx.AttributeProto.verify|verify} messages.
             * @function encode
             * @memberof onnx.AttributeProto
             * @static
             * @param {onnx.IAttributeProto} message AttributeProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AttributeProto.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.name != null && message.hasOwnProperty("name"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
                if (message.f != null && message.hasOwnProperty("f"))
                    writer.uint32(/* id 2, wireType 5 =*/21).float(message.f);
                if (message.i != null && message.hasOwnProperty("i"))
                    writer.uint32(/* id 3, wireType 0 =*/24).int64(message.i);
                if (message.s != null && message.hasOwnProperty("s"))
                    writer.uint32(/* id 4, wireType 2 =*/34).bytes(message.s);
                if (message.t != null && message.hasOwnProperty("t"))
                    $root.onnx.TensorProto.encode(message.t, writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
                if (message.g != null && message.hasOwnProperty("g"))
                    $root.onnx.GraphProto.encode(message.g, writer.uint32(/* id 6, wireType 2 =*/50).fork()).ldelim();
                if (message.floats != null && message.floats.length)
                    for (var i = 0; i < message.floats.length; ++i)
                        writer.uint32(/* id 7, wireType 5 =*/61).float(message.floats[i]);
                if (message.ints != null && message.ints.length)
                    for (var i = 0; i < message.ints.length; ++i)
                        writer.uint32(/* id 8, wireType 0 =*/64).int64(message.ints[i]);
                if (message.strings != null && message.strings.length)
                    for (var i = 0; i < message.strings.length; ++i)
                        writer.uint32(/* id 9, wireType 2 =*/74).bytes(message.strings[i]);
                if (message.tensors != null && message.tensors.length)
                    for (var i = 0; i < message.tensors.length; ++i)
                        $root.onnx.TensorProto.encode(message.tensors[i], writer.uint32(/* id 10, wireType 2 =*/82).fork()).ldelim();
                if (message.graphs != null && message.graphs.length)
                    for (var i = 0; i < message.graphs.length; ++i)
                        $root.onnx.GraphProto.encode(message.graphs[i], writer.uint32(/* id 11, wireType 2 =*/90).fork()).ldelim();
                if (message.docString != null && message.hasOwnProperty("docString"))
                    writer.uint32(/* id 13, wireType 2 =*/106).string(message.docString);
                if (message.type != null && message.hasOwnProperty("type"))
                    writer.uint32(/* id 20, wireType 0 =*/160).int32(message.type);
                return writer;
            };
    
            /**
             * Encodes the specified AttributeProto message, length delimited. Does not implicitly {@link onnx.AttributeProto.verify|verify} messages.
             * @function encodeDelimited
             * @memberof onnx.AttributeProto
             * @static
             * @param {onnx.IAttributeProto} message AttributeProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AttributeProto.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes an AttributeProto message from the specified reader or buffer.
             * @function decode
             * @memberof onnx.AttributeProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {onnx.AttributeProto} AttributeProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
                    case 13:
                        message.docString = reader.string();
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
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            /**
             * Decodes an AttributeProto message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof onnx.AttributeProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {onnx.AttributeProto} AttributeProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AttributeProto.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies an AttributeProto message.
             * @function verify
             * @memberof onnx.AttributeProto
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            AttributeProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.docString != null && message.hasOwnProperty("docString"))
                    if (!$util.isString(message.docString))
                        return "docString: string expected";
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
                        break;
                    }
                if (message.f != null && message.hasOwnProperty("f"))
                    if (typeof message.f !== "number")
                        return "f: number expected";
                if (message.i != null && message.hasOwnProperty("i"))
                    if (!$util.isInteger(message.i) && !(message.i && $util.isInteger(message.i.low) && $util.isInteger(message.i.high)))
                        return "i: integer|Long expected";
                if (message.s != null && message.hasOwnProperty("s"))
                    if (!(message.s && typeof message.s.length === "number" || $util.isString(message.s)))
                        return "s: buffer expected";
                if (message.t != null && message.hasOwnProperty("t")) {
                    var error = $root.onnx.TensorProto.verify(message.t);
                    if (error)
                        return "t." + error;
                }
                if (message.g != null && message.hasOwnProperty("g")) {
                    var error = $root.onnx.GraphProto.verify(message.g);
                    if (error)
                        return "g." + error;
                }
                if (message.floats != null && message.hasOwnProperty("floats")) {
                    if (!Array.isArray(message.floats))
                        return "floats: array expected";
                    for (var i = 0; i < message.floats.length; ++i)
                        if (typeof message.floats[i] !== "number")
                            return "floats: number[] expected";
                }
                if (message.ints != null && message.hasOwnProperty("ints")) {
                    if (!Array.isArray(message.ints))
                        return "ints: array expected";
                    for (var i = 0; i < message.ints.length; ++i)
                        if (!$util.isInteger(message.ints[i]) && !(message.ints[i] && $util.isInteger(message.ints[i].low) && $util.isInteger(message.ints[i].high)))
                            return "ints: integer|Long[] expected";
                }
                if (message.strings != null && message.hasOwnProperty("strings")) {
                    if (!Array.isArray(message.strings))
                        return "strings: array expected";
                    for (var i = 0; i < message.strings.length; ++i)
                        if (!(message.strings[i] && typeof message.strings[i].length === "number" || $util.isString(message.strings[i])))
                            return "strings: buffer[] expected";
                }
                if (message.tensors != null && message.hasOwnProperty("tensors")) {
                    if (!Array.isArray(message.tensors))
                        return "tensors: array expected";
                    for (var i = 0; i < message.tensors.length; ++i) {
                        var error = $root.onnx.TensorProto.verify(message.tensors[i]);
                        if (error)
                            return "tensors." + error;
                    }
                }
                if (message.graphs != null && message.hasOwnProperty("graphs")) {
                    if (!Array.isArray(message.graphs))
                        return "graphs: array expected";
                    for (var i = 0; i < message.graphs.length; ++i) {
                        var error = $root.onnx.GraphProto.verify(message.graphs[i]);
                        if (error)
                            return "graphs." + error;
                    }
                }
                return null;
            };
    
            /**
             * Creates an AttributeProto message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof onnx.AttributeProto
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {onnx.AttributeProto} AttributeProto
             */
            AttributeProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.AttributeProto)
                    return object;
                var message = new $root.onnx.AttributeProto();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.docString != null)
                    message.docString = String(object.docString);
                switch (object.type) {
                case "UNDEFINED":
                case 0:
                    message.type = 0;
                    break;
                case "FLOAT":
                case 1:
                    message.type = 1;
                    break;
                case "INT":
                case 2:
                    message.type = 2;
                    break;
                case "STRING":
                case 3:
                    message.type = 3;
                    break;
                case "TENSOR":
                case 4:
                    message.type = 4;
                    break;
                case "GRAPH":
                case 5:
                    message.type = 5;
                    break;
                case "FLOATS":
                case 6:
                    message.type = 6;
                    break;
                case "INTS":
                case 7:
                    message.type = 7;
                    break;
                case "STRINGS":
                case 8:
                    message.type = 8;
                    break;
                case "TENSORS":
                case 9:
                    message.type = 9;
                    break;
                case "GRAPHS":
                case 10:
                    message.type = 10;
                    break;
                }
                if (object.f != null)
                    message.f = Number(object.f);
                if (object.i != null)
                    if ($util.Long)
                        (message.i = $util.Long.fromValue(object.i)).unsigned = false;
                    else if (typeof object.i === "string")
                        message.i = parseInt(object.i, 10);
                    else if (typeof object.i === "number")
                        message.i = object.i;
                    else if (typeof object.i === "object")
                        message.i = new $util.LongBits(object.i.low >>> 0, object.i.high >>> 0).toNumber();
                if (object.s != null)
                    if (typeof object.s === "string")
                        $util.base64.decode(object.s, message.s = $util.newBuffer($util.base64.length(object.s)), 0);
                    else if (object.s.length)
                        message.s = object.s;
                if (object.t != null) {
                    if (typeof object.t !== "object")
                        throw TypeError(".onnx.AttributeProto.t: object expected");
                    message.t = $root.onnx.TensorProto.fromObject(object.t);
                }
                if (object.g != null) {
                    if (typeof object.g !== "object")
                        throw TypeError(".onnx.AttributeProto.g: object expected");
                    message.g = $root.onnx.GraphProto.fromObject(object.g);
                }
                if (object.floats) {
                    if (!Array.isArray(object.floats))
                        throw TypeError(".onnx.AttributeProto.floats: array expected");
                    message.floats = [];
                    for (var i = 0; i < object.floats.length; ++i)
                        message.floats[i] = Number(object.floats[i]);
                }
                if (object.ints) {
                    if (!Array.isArray(object.ints))
                        throw TypeError(".onnx.AttributeProto.ints: array expected");
                    message.ints = [];
                    for (var i = 0; i < object.ints.length; ++i)
                        if ($util.Long)
                            (message.ints[i] = $util.Long.fromValue(object.ints[i])).unsigned = false;
                        else if (typeof object.ints[i] === "string")
                            message.ints[i] = parseInt(object.ints[i], 10);
                        else if (typeof object.ints[i] === "number")
                            message.ints[i] = object.ints[i];
                        else if (typeof object.ints[i] === "object")
                            message.ints[i] = new $util.LongBits(object.ints[i].low >>> 0, object.ints[i].high >>> 0).toNumber();
                }
                if (object.strings) {
                    if (!Array.isArray(object.strings))
                        throw TypeError(".onnx.AttributeProto.strings: array expected");
                    message.strings = [];
                    for (var i = 0; i < object.strings.length; ++i)
                        if (typeof object.strings[i] === "string")
                            $util.base64.decode(object.strings[i], message.strings[i] = $util.newBuffer($util.base64.length(object.strings[i])), 0);
                        else if (object.strings[i].length)
                            message.strings[i] = object.strings[i];
                }
                if (object.tensors) {
                    if (!Array.isArray(object.tensors))
                        throw TypeError(".onnx.AttributeProto.tensors: array expected");
                    message.tensors = [];
                    for (var i = 0; i < object.tensors.length; ++i) {
                        if (typeof object.tensors[i] !== "object")
                            throw TypeError(".onnx.AttributeProto.tensors: object expected");
                        message.tensors[i] = $root.onnx.TensorProto.fromObject(object.tensors[i]);
                    }
                }
                if (object.graphs) {
                    if (!Array.isArray(object.graphs))
                        throw TypeError(".onnx.AttributeProto.graphs: array expected");
                    message.graphs = [];
                    for (var i = 0; i < object.graphs.length; ++i) {
                        if (typeof object.graphs[i] !== "object")
                            throw TypeError(".onnx.AttributeProto.graphs: object expected");
                        message.graphs[i] = $root.onnx.GraphProto.fromObject(object.graphs[i]);
                    }
                }
                return message;
            };
    
            /**
             * Creates a plain object from an AttributeProto message. Also converts values to other types if specified.
             * @function toObject
             * @memberof onnx.AttributeProto
             * @static
             * @param {onnx.AttributeProto} message AttributeProto
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            AttributeProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.floats = [];
                    object.ints = [];
                    object.strings = [];
                    object.tensors = [];
                    object.graphs = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.f = 0;
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.i = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.i = options.longs === String ? "0" : 0;
                    object.s = options.bytes === String ? "" : [];
                    object.t = null;
                    object.g = null;
                    object.docString = "";
                    object.type = options.enums === String ? "UNDEFINED" : 0;
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.f != null && message.hasOwnProperty("f"))
                    object.f = options.json && !isFinite(message.f) ? String(message.f) : message.f;
                if (message.i != null && message.hasOwnProperty("i"))
                    if (typeof message.i === "number")
                        object.i = options.longs === String ? String(message.i) : message.i;
                    else
                        object.i = options.longs === String ? $util.Long.prototype.toString.call(message.i) : options.longs === Number ? new $util.LongBits(message.i.low >>> 0, message.i.high >>> 0).toNumber() : message.i;
                if (message.s != null && message.hasOwnProperty("s"))
                    object.s = options.bytes === String ? $util.base64.encode(message.s, 0, message.s.length) : options.bytes === Array ? Array.prototype.slice.call(message.s) : message.s;
                if (message.t != null && message.hasOwnProperty("t"))
                    object.t = $root.onnx.TensorProto.toObject(message.t, options);
                if (message.g != null && message.hasOwnProperty("g"))
                    object.g = $root.onnx.GraphProto.toObject(message.g, options);
                if (message.floats && message.floats.length) {
                    object.floats = [];
                    for (var j = 0; j < message.floats.length; ++j)
                        object.floats[j] = options.json && !isFinite(message.floats[j]) ? String(message.floats[j]) : message.floats[j];
                }
                if (message.ints && message.ints.length) {
                    object.ints = [];
                    for (var j = 0; j < message.ints.length; ++j)
                        if (typeof message.ints[j] === "number")
                            object.ints[j] = options.longs === String ? String(message.ints[j]) : message.ints[j];
                        else
                            object.ints[j] = options.longs === String ? $util.Long.prototype.toString.call(message.ints[j]) : options.longs === Number ? new $util.LongBits(message.ints[j].low >>> 0, message.ints[j].high >>> 0).toNumber() : message.ints[j];
                }
                if (message.strings && message.strings.length) {
                    object.strings = [];
                    for (var j = 0; j < message.strings.length; ++j)
                        object.strings[j] = options.bytes === String ? $util.base64.encode(message.strings[j], 0, message.strings[j].length) : options.bytes === Array ? Array.prototype.slice.call(message.strings[j]) : message.strings[j];
                }
                if (message.tensors && message.tensors.length) {
                    object.tensors = [];
                    for (var j = 0; j < message.tensors.length; ++j)
                        object.tensors[j] = $root.onnx.TensorProto.toObject(message.tensors[j], options);
                }
                if (message.graphs && message.graphs.length) {
                    object.graphs = [];
                    for (var j = 0; j < message.graphs.length; ++j)
                        object.graphs[j] = $root.onnx.GraphProto.toObject(message.graphs[j], options);
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    object.docString = message.docString;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = options.enums === String ? $root.onnx.AttributeProto.AttributeType[message.type] : message.type;
                return object;
            };
    
            /**
             * Converts this AttributeProto to JSON.
             * @function toJSON
             * @memberof onnx.AttributeProto
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            AttributeProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            /**
             * AttributeType enum.
             * @name onnx.AttributeProto.AttributeType
             * @enum {string}
             * @property {number} UNDEFINED=0 UNDEFINED value
             * @property {number} FLOAT=1 FLOAT value
             * @property {number} INT=2 INT value
             * @property {number} STRING=3 STRING value
             * @property {number} TENSOR=4 TENSOR value
             * @property {number} GRAPH=5 GRAPH value
             * @property {number} FLOATS=6 FLOATS value
             * @property {number} INTS=7 INTS value
             * @property {number} STRINGS=8 STRINGS value
             * @property {number} TENSORS=9 TENSORS value
             * @property {number} GRAPHS=10 GRAPHS value
             */
            AttributeProto.AttributeType = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "UNDEFINED"] = 0;
                values[valuesById[1] = "FLOAT"] = 1;
                values[valuesById[2] = "INT"] = 2;
                values[valuesById[3] = "STRING"] = 3;
                values[valuesById[4] = "TENSOR"] = 4;
                values[valuesById[5] = "GRAPH"] = 5;
                values[valuesById[6] = "FLOATS"] = 6;
                values[valuesById[7] = "INTS"] = 7;
                values[valuesById[8] = "STRINGS"] = 8;
                values[valuesById[9] = "TENSORS"] = 9;
                values[valuesById[10] = "GRAPHS"] = 10;
                return values;
            })();
    
            return AttributeProto;
        })();
    
        onnx.ValueInfoProto = (function() {
    
            /**
             * Properties of a ValueInfoProto.
             * @memberof onnx
             * @interface IValueInfoProto
             * @property {string|null} [name] ValueInfoProto name
             * @property {onnx.ITypeProto|null} [type] ValueInfoProto type
             * @property {string|null} [docString] ValueInfoProto docString
             */
    
            /**
             * Constructs a new ValueInfoProto.
             * @memberof onnx
             * @classdesc Represents a ValueInfoProto.
             * @implements IValueInfoProto
             * @constructor
             * @param {onnx.IValueInfoProto=} [properties] Properties to set
             */
            function ValueInfoProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * ValueInfoProto name.
             * @member {string} name
             * @memberof onnx.ValueInfoProto
             * @instance
             */
            ValueInfoProto.prototype.name = "";
    
            /**
             * ValueInfoProto type.
             * @member {onnx.ITypeProto|null|undefined} type
             * @memberof onnx.ValueInfoProto
             * @instance
             */
            ValueInfoProto.prototype.type = null;
    
            /**
             * ValueInfoProto docString.
             * @member {string} docString
             * @memberof onnx.ValueInfoProto
             * @instance
             */
            ValueInfoProto.prototype.docString = "";
    
            /**
             * Creates a new ValueInfoProto instance using the specified properties.
             * @function create
             * @memberof onnx.ValueInfoProto
             * @static
             * @param {onnx.IValueInfoProto=} [properties] Properties to set
             * @returns {onnx.ValueInfoProto} ValueInfoProto instance
             */
            ValueInfoProto.create = function create(properties) {
                return new ValueInfoProto(properties);
            };
    
            /**
             * Encodes the specified ValueInfoProto message. Does not implicitly {@link onnx.ValueInfoProto.verify|verify} messages.
             * @function encode
             * @memberof onnx.ValueInfoProto
             * @static
             * @param {onnx.IValueInfoProto} message ValueInfoProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ValueInfoProto.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.name != null && message.hasOwnProperty("name"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
                if (message.type != null && message.hasOwnProperty("type"))
                    $root.onnx.TypeProto.encode(message.type, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                if (message.docString != null && message.hasOwnProperty("docString"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.docString);
                return writer;
            };
    
            /**
             * Encodes the specified ValueInfoProto message, length delimited. Does not implicitly {@link onnx.ValueInfoProto.verify|verify} messages.
             * @function encodeDelimited
             * @memberof onnx.ValueInfoProto
             * @static
             * @param {onnx.IValueInfoProto} message ValueInfoProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ValueInfoProto.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a ValueInfoProto message from the specified reader or buffer.
             * @function decode
             * @memberof onnx.ValueInfoProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {onnx.ValueInfoProto} ValueInfoProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
                        message.docString = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            /**
             * Decodes a ValueInfoProto message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof onnx.ValueInfoProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {onnx.ValueInfoProto} ValueInfoProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ValueInfoProto.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a ValueInfoProto message.
             * @function verify
             * @memberof onnx.ValueInfoProto
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            ValueInfoProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.type != null && message.hasOwnProperty("type")) {
                    var error = $root.onnx.TypeProto.verify(message.type);
                    if (error)
                        return "type." + error;
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    if (!$util.isString(message.docString))
                        return "docString: string expected";
                return null;
            };
    
            /**
             * Creates a ValueInfoProto message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof onnx.ValueInfoProto
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {onnx.ValueInfoProto} ValueInfoProto
             */
            ValueInfoProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.ValueInfoProto)
                    return object;
                var message = new $root.onnx.ValueInfoProto();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.type != null) {
                    if (typeof object.type !== "object")
                        throw TypeError(".onnx.ValueInfoProto.type: object expected");
                    message.type = $root.onnx.TypeProto.fromObject(object.type);
                }
                if (object.docString != null)
                    message.docString = String(object.docString);
                return message;
            };
    
            /**
             * Creates a plain object from a ValueInfoProto message. Also converts values to other types if specified.
             * @function toObject
             * @memberof onnx.ValueInfoProto
             * @static
             * @param {onnx.ValueInfoProto} message ValueInfoProto
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            ValueInfoProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.name = "";
                    object.type = null;
                    object.docString = "";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = $root.onnx.TypeProto.toObject(message.type, options);
                if (message.docString != null && message.hasOwnProperty("docString"))
                    object.docString = message.docString;
                return object;
            };
    
            /**
             * Converts this ValueInfoProto to JSON.
             * @function toJSON
             * @memberof onnx.ValueInfoProto
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            ValueInfoProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ValueInfoProto;
        })();
    
        onnx.NodeProto = (function() {
    
            /**
             * Properties of a NodeProto.
             * @memberof onnx
             * @interface INodeProto
             * @property {Array.<string>|null} [input] NodeProto input
             * @property {Array.<string>|null} [output] NodeProto output
             * @property {string|null} [name] NodeProto name
             * @property {string|null} [opType] NodeProto opType
             * @property {string|null} [domain] NodeProto domain
             * @property {Array.<onnx.IAttributeProto>|null} [attribute] NodeProto attribute
             * @property {string|null} [docString] NodeProto docString
             */
    
            /**
             * Constructs a new NodeProto.
             * @memberof onnx
             * @classdesc Represents a NodeProto.
             * @implements INodeProto
             * @constructor
             * @param {onnx.INodeProto=} [properties] Properties to set
             */
            function NodeProto(properties) {
                this.input = [];
                this.output = [];
                this.attribute = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * NodeProto input.
             * @member {Array.<string>} input
             * @memberof onnx.NodeProto
             * @instance
             */
            NodeProto.prototype.input = $util.emptyArray;
    
            /**
             * NodeProto output.
             * @member {Array.<string>} output
             * @memberof onnx.NodeProto
             * @instance
             */
            NodeProto.prototype.output = $util.emptyArray;
    
            /**
             * NodeProto name.
             * @member {string} name
             * @memberof onnx.NodeProto
             * @instance
             */
            NodeProto.prototype.name = "";
    
            /**
             * NodeProto opType.
             * @member {string} opType
             * @memberof onnx.NodeProto
             * @instance
             */
            NodeProto.prototype.opType = "";
    
            /**
             * NodeProto domain.
             * @member {string} domain
             * @memberof onnx.NodeProto
             * @instance
             */
            NodeProto.prototype.domain = "";
    
            /**
             * NodeProto attribute.
             * @member {Array.<onnx.IAttributeProto>} attribute
             * @memberof onnx.NodeProto
             * @instance
             */
            NodeProto.prototype.attribute = $util.emptyArray;
    
            /**
             * NodeProto docString.
             * @member {string} docString
             * @memberof onnx.NodeProto
             * @instance
             */
            NodeProto.prototype.docString = "";
    
            /**
             * Creates a new NodeProto instance using the specified properties.
             * @function create
             * @memberof onnx.NodeProto
             * @static
             * @param {onnx.INodeProto=} [properties] Properties to set
             * @returns {onnx.NodeProto} NodeProto instance
             */
            NodeProto.create = function create(properties) {
                return new NodeProto(properties);
            };
    
            /**
             * Encodes the specified NodeProto message. Does not implicitly {@link onnx.NodeProto.verify|verify} messages.
             * @function encode
             * @memberof onnx.NodeProto
             * @static
             * @param {onnx.INodeProto} message NodeProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            NodeProto.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.input != null && message.input.length)
                    for (var i = 0; i < message.input.length; ++i)
                        writer.uint32(/* id 1, wireType 2 =*/10).string(message.input[i]);
                if (message.output != null && message.output.length)
                    for (var i = 0; i < message.output.length; ++i)
                        writer.uint32(/* id 2, wireType 2 =*/18).string(message.output[i]);
                if (message.name != null && message.hasOwnProperty("name"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.name);
                if (message.opType != null && message.hasOwnProperty("opType"))
                    writer.uint32(/* id 4, wireType 2 =*/34).string(message.opType);
                if (message.attribute != null && message.attribute.length)
                    for (var i = 0; i < message.attribute.length; ++i)
                        $root.onnx.AttributeProto.encode(message.attribute[i], writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
                if (message.docString != null && message.hasOwnProperty("docString"))
                    writer.uint32(/* id 6, wireType 2 =*/50).string(message.docString);
                if (message.domain != null && message.hasOwnProperty("domain"))
                    writer.uint32(/* id 7, wireType 2 =*/58).string(message.domain);
                return writer;
            };
    
            /**
             * Encodes the specified NodeProto message, length delimited. Does not implicitly {@link onnx.NodeProto.verify|verify} messages.
             * @function encodeDelimited
             * @memberof onnx.NodeProto
             * @static
             * @param {onnx.INodeProto} message NodeProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            NodeProto.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a NodeProto message from the specified reader or buffer.
             * @function decode
             * @memberof onnx.NodeProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {onnx.NodeProto} NodeProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
                        message.opType = reader.string();
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
                        message.docString = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            /**
             * Decodes a NodeProto message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof onnx.NodeProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {onnx.NodeProto} NodeProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            NodeProto.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a NodeProto message.
             * @function verify
             * @memberof onnx.NodeProto
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            NodeProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.input != null && message.hasOwnProperty("input")) {
                    if (!Array.isArray(message.input))
                        return "input: array expected";
                    for (var i = 0; i < message.input.length; ++i)
                        if (!$util.isString(message.input[i]))
                            return "input: string[] expected";
                }
                if (message.output != null && message.hasOwnProperty("output")) {
                    if (!Array.isArray(message.output))
                        return "output: array expected";
                    for (var i = 0; i < message.output.length; ++i)
                        if (!$util.isString(message.output[i]))
                            return "output: string[] expected";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.opType != null && message.hasOwnProperty("opType"))
                    if (!$util.isString(message.opType))
                        return "opType: string expected";
                if (message.domain != null && message.hasOwnProperty("domain"))
                    if (!$util.isString(message.domain))
                        return "domain: string expected";
                if (message.attribute != null && message.hasOwnProperty("attribute")) {
                    if (!Array.isArray(message.attribute))
                        return "attribute: array expected";
                    for (var i = 0; i < message.attribute.length; ++i) {
                        var error = $root.onnx.AttributeProto.verify(message.attribute[i]);
                        if (error)
                            return "attribute." + error;
                    }
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    if (!$util.isString(message.docString))
                        return "docString: string expected";
                return null;
            };
    
            /**
             * Creates a NodeProto message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof onnx.NodeProto
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {onnx.NodeProto} NodeProto
             */
            NodeProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.NodeProto)
                    return object;
                var message = new $root.onnx.NodeProto();
                if (object.input) {
                    if (!Array.isArray(object.input))
                        throw TypeError(".onnx.NodeProto.input: array expected");
                    message.input = [];
                    for (var i = 0; i < object.input.length; ++i)
                        message.input[i] = String(object.input[i]);
                }
                if (object.output) {
                    if (!Array.isArray(object.output))
                        throw TypeError(".onnx.NodeProto.output: array expected");
                    message.output = [];
                    for (var i = 0; i < object.output.length; ++i)
                        message.output[i] = String(object.output[i]);
                }
                if (object.name != null)
                    message.name = String(object.name);
                if (object.opType != null)
                    message.opType = String(object.opType);
                if (object.domain != null)
                    message.domain = String(object.domain);
                if (object.attribute) {
                    if (!Array.isArray(object.attribute))
                        throw TypeError(".onnx.NodeProto.attribute: array expected");
                    message.attribute = [];
                    for (var i = 0; i < object.attribute.length; ++i) {
                        if (typeof object.attribute[i] !== "object")
                            throw TypeError(".onnx.NodeProto.attribute: object expected");
                        message.attribute[i] = $root.onnx.AttributeProto.fromObject(object.attribute[i]);
                    }
                }
                if (object.docString != null)
                    message.docString = String(object.docString);
                return message;
            };
    
            /**
             * Creates a plain object from a NodeProto message. Also converts values to other types if specified.
             * @function toObject
             * @memberof onnx.NodeProto
             * @static
             * @param {onnx.NodeProto} message NodeProto
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            NodeProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.input = [];
                    object.output = [];
                    object.attribute = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.opType = "";
                    object.docString = "";
                    object.domain = "";
                }
                if (message.input && message.input.length) {
                    object.input = [];
                    for (var j = 0; j < message.input.length; ++j)
                        object.input[j] = message.input[j];
                }
                if (message.output && message.output.length) {
                    object.output = [];
                    for (var j = 0; j < message.output.length; ++j)
                        object.output[j] = message.output[j];
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.opType != null && message.hasOwnProperty("opType"))
                    object.opType = message.opType;
                if (message.attribute && message.attribute.length) {
                    object.attribute = [];
                    for (var j = 0; j < message.attribute.length; ++j)
                        object.attribute[j] = $root.onnx.AttributeProto.toObject(message.attribute[j], options);
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    object.docString = message.docString;
                if (message.domain != null && message.hasOwnProperty("domain"))
                    object.domain = message.domain;
                return object;
            };
    
            /**
             * Converts this NodeProto to JSON.
             * @function toJSON
             * @memberof onnx.NodeProto
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            NodeProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return NodeProto;
        })();
    
        onnx.ModelProto = (function() {
    
            /**
             * Properties of a ModelProto.
             * @memberof onnx
             * @interface IModelProto
             * @property {number|Long|null} [irVersion] ModelProto irVersion
             * @property {Array.<onnx.IOperatorSetIdProto>|null} [opsetImport] ModelProto opsetImport
             * @property {string|null} [producerName] ModelProto producerName
             * @property {string|null} [producerVersion] ModelProto producerVersion
             * @property {string|null} [domain] ModelProto domain
             * @property {number|Long|null} [modelVersion] ModelProto modelVersion
             * @property {string|null} [docString] ModelProto docString
             * @property {onnx.IGraphProto|null} [graph] ModelProto graph
             * @property {Array.<onnx.IStringStringEntryProto>|null} [metadataProps] ModelProto metadataProps
             */
    
            /**
             * Constructs a new ModelProto.
             * @memberof onnx
             * @classdesc Represents a ModelProto.
             * @implements IModelProto
             * @constructor
             * @param {onnx.IModelProto=} [properties] Properties to set
             */
            function ModelProto(properties) {
                this.opsetImport = [];
                this.metadataProps = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * ModelProto irVersion.
             * @member {number|Long} irVersion
             * @memberof onnx.ModelProto
             * @instance
             */
            ModelProto.prototype.irVersion = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
            /**
             * ModelProto opsetImport.
             * @member {Array.<onnx.IOperatorSetIdProto>} opsetImport
             * @memberof onnx.ModelProto
             * @instance
             */
            ModelProto.prototype.opsetImport = $util.emptyArray;
    
            /**
             * ModelProto producerName.
             * @member {string} producerName
             * @memberof onnx.ModelProto
             * @instance
             */
            ModelProto.prototype.producerName = "";
    
            /**
             * ModelProto producerVersion.
             * @member {string} producerVersion
             * @memberof onnx.ModelProto
             * @instance
             */
            ModelProto.prototype.producerVersion = "";
    
            /**
             * ModelProto domain.
             * @member {string} domain
             * @memberof onnx.ModelProto
             * @instance
             */
            ModelProto.prototype.domain = "";
    
            /**
             * ModelProto modelVersion.
             * @member {number|Long} modelVersion
             * @memberof onnx.ModelProto
             * @instance
             */
            ModelProto.prototype.modelVersion = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
            /**
             * ModelProto docString.
             * @member {string} docString
             * @memberof onnx.ModelProto
             * @instance
             */
            ModelProto.prototype.docString = "";
    
            /**
             * ModelProto graph.
             * @member {onnx.IGraphProto|null|undefined} graph
             * @memberof onnx.ModelProto
             * @instance
             */
            ModelProto.prototype.graph = null;
    
            /**
             * ModelProto metadataProps.
             * @member {Array.<onnx.IStringStringEntryProto>} metadataProps
             * @memberof onnx.ModelProto
             * @instance
             */
            ModelProto.prototype.metadataProps = $util.emptyArray;
    
            /**
             * Creates a new ModelProto instance using the specified properties.
             * @function create
             * @memberof onnx.ModelProto
             * @static
             * @param {onnx.IModelProto=} [properties] Properties to set
             * @returns {onnx.ModelProto} ModelProto instance
             */
            ModelProto.create = function create(properties) {
                return new ModelProto(properties);
            };
    
            /**
             * Encodes the specified ModelProto message. Does not implicitly {@link onnx.ModelProto.verify|verify} messages.
             * @function encode
             * @memberof onnx.ModelProto
             * @static
             * @param {onnx.IModelProto} message ModelProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ModelProto.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.irVersion != null && message.hasOwnProperty("irVersion"))
                    writer.uint32(/* id 1, wireType 0 =*/8).int64(message.irVersion);
                if (message.producerName != null && message.hasOwnProperty("producerName"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.producerName);
                if (message.producerVersion != null && message.hasOwnProperty("producerVersion"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.producerVersion);
                if (message.domain != null && message.hasOwnProperty("domain"))
                    writer.uint32(/* id 4, wireType 2 =*/34).string(message.domain);
                if (message.modelVersion != null && message.hasOwnProperty("modelVersion"))
                    writer.uint32(/* id 5, wireType 0 =*/40).int64(message.modelVersion);
                if (message.docString != null && message.hasOwnProperty("docString"))
                    writer.uint32(/* id 6, wireType 2 =*/50).string(message.docString);
                if (message.graph != null && message.hasOwnProperty("graph"))
                    $root.onnx.GraphProto.encode(message.graph, writer.uint32(/* id 7, wireType 2 =*/58).fork()).ldelim();
                if (message.opsetImport != null && message.opsetImport.length)
                    for (var i = 0; i < message.opsetImport.length; ++i)
                        $root.onnx.OperatorSetIdProto.encode(message.opsetImport[i], writer.uint32(/* id 8, wireType 2 =*/66).fork()).ldelim();
                if (message.metadataProps != null && message.metadataProps.length)
                    for (var i = 0; i < message.metadataProps.length; ++i)
                        $root.onnx.StringStringEntryProto.encode(message.metadataProps[i], writer.uint32(/* id 14, wireType 2 =*/114).fork()).ldelim();
                return writer;
            };
    
            /**
             * Encodes the specified ModelProto message, length delimited. Does not implicitly {@link onnx.ModelProto.verify|verify} messages.
             * @function encodeDelimited
             * @memberof onnx.ModelProto
             * @static
             * @param {onnx.IModelProto} message ModelProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ModelProto.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a ModelProto message from the specified reader or buffer.
             * @function decode
             * @memberof onnx.ModelProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {onnx.ModelProto} ModelProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ModelProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.ModelProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.irVersion = reader.int64();
                        break;
                    case 8:
                        if (!(message.opsetImport && message.opsetImport.length))
                            message.opsetImport = [];
                        message.opsetImport.push($root.onnx.OperatorSetIdProto.decode(reader, reader.uint32()));
                        break;
                    case 2:
                        message.producerName = reader.string();
                        break;
                    case 3:
                        message.producerVersion = reader.string();
                        break;
                    case 4:
                        message.domain = reader.string();
                        break;
                    case 5:
                        message.modelVersion = reader.int64();
                        break;
                    case 6:
                        message.docString = reader.string();
                        break;
                    case 7:
                        message.graph = $root.onnx.GraphProto.decode(reader, reader.uint32());
                        break;
                    case 14:
                        if (!(message.metadataProps && message.metadataProps.length))
                            message.metadataProps = [];
                        message.metadataProps.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            /**
             * Decodes a ModelProto message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof onnx.ModelProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {onnx.ModelProto} ModelProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ModelProto.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a ModelProto message.
             * @function verify
             * @memberof onnx.ModelProto
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            ModelProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.irVersion != null && message.hasOwnProperty("irVersion"))
                    if (!$util.isInteger(message.irVersion) && !(message.irVersion && $util.isInteger(message.irVersion.low) && $util.isInteger(message.irVersion.high)))
                        return "irVersion: integer|Long expected";
                if (message.opsetImport != null && message.hasOwnProperty("opsetImport")) {
                    if (!Array.isArray(message.opsetImport))
                        return "opsetImport: array expected";
                    for (var i = 0; i < message.opsetImport.length; ++i) {
                        var error = $root.onnx.OperatorSetIdProto.verify(message.opsetImport[i]);
                        if (error)
                            return "opsetImport." + error;
                    }
                }
                if (message.producerName != null && message.hasOwnProperty("producerName"))
                    if (!$util.isString(message.producerName))
                        return "producerName: string expected";
                if (message.producerVersion != null && message.hasOwnProperty("producerVersion"))
                    if (!$util.isString(message.producerVersion))
                        return "producerVersion: string expected";
                if (message.domain != null && message.hasOwnProperty("domain"))
                    if (!$util.isString(message.domain))
                        return "domain: string expected";
                if (message.modelVersion != null && message.hasOwnProperty("modelVersion"))
                    if (!$util.isInteger(message.modelVersion) && !(message.modelVersion && $util.isInteger(message.modelVersion.low) && $util.isInteger(message.modelVersion.high)))
                        return "modelVersion: integer|Long expected";
                if (message.docString != null && message.hasOwnProperty("docString"))
                    if (!$util.isString(message.docString))
                        return "docString: string expected";
                if (message.graph != null && message.hasOwnProperty("graph")) {
                    var error = $root.onnx.GraphProto.verify(message.graph);
                    if (error)
                        return "graph." + error;
                }
                if (message.metadataProps != null && message.hasOwnProperty("metadataProps")) {
                    if (!Array.isArray(message.metadataProps))
                        return "metadataProps: array expected";
                    for (var i = 0; i < message.metadataProps.length; ++i) {
                        var error = $root.onnx.StringStringEntryProto.verify(message.metadataProps[i]);
                        if (error)
                            return "metadataProps." + error;
                    }
                }
                return null;
            };
    
            /**
             * Creates a ModelProto message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof onnx.ModelProto
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {onnx.ModelProto} ModelProto
             */
            ModelProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.ModelProto)
                    return object;
                var message = new $root.onnx.ModelProto();
                if (object.irVersion != null)
                    if ($util.Long)
                        (message.irVersion = $util.Long.fromValue(object.irVersion)).unsigned = false;
                    else if (typeof object.irVersion === "string")
                        message.irVersion = parseInt(object.irVersion, 10);
                    else if (typeof object.irVersion === "number")
                        message.irVersion = object.irVersion;
                    else if (typeof object.irVersion === "object")
                        message.irVersion = new $util.LongBits(object.irVersion.low >>> 0, object.irVersion.high >>> 0).toNumber();
                if (object.opsetImport) {
                    if (!Array.isArray(object.opsetImport))
                        throw TypeError(".onnx.ModelProto.opsetImport: array expected");
                    message.opsetImport = [];
                    for (var i = 0; i < object.opsetImport.length; ++i) {
                        if (typeof object.opsetImport[i] !== "object")
                            throw TypeError(".onnx.ModelProto.opsetImport: object expected");
                        message.opsetImport[i] = $root.onnx.OperatorSetIdProto.fromObject(object.opsetImport[i]);
                    }
                }
                if (object.producerName != null)
                    message.producerName = String(object.producerName);
                if (object.producerVersion != null)
                    message.producerVersion = String(object.producerVersion);
                if (object.domain != null)
                    message.domain = String(object.domain);
                if (object.modelVersion != null)
                    if ($util.Long)
                        (message.modelVersion = $util.Long.fromValue(object.modelVersion)).unsigned = false;
                    else if (typeof object.modelVersion === "string")
                        message.modelVersion = parseInt(object.modelVersion, 10);
                    else if (typeof object.modelVersion === "number")
                        message.modelVersion = object.modelVersion;
                    else if (typeof object.modelVersion === "object")
                        message.modelVersion = new $util.LongBits(object.modelVersion.low >>> 0, object.modelVersion.high >>> 0).toNumber();
                if (object.docString != null)
                    message.docString = String(object.docString);
                if (object.graph != null) {
                    if (typeof object.graph !== "object")
                        throw TypeError(".onnx.ModelProto.graph: object expected");
                    message.graph = $root.onnx.GraphProto.fromObject(object.graph);
                }
                if (object.metadataProps) {
                    if (!Array.isArray(object.metadataProps))
                        throw TypeError(".onnx.ModelProto.metadataProps: array expected");
                    message.metadataProps = [];
                    for (var i = 0; i < object.metadataProps.length; ++i) {
                        if (typeof object.metadataProps[i] !== "object")
                            throw TypeError(".onnx.ModelProto.metadataProps: object expected");
                        message.metadataProps[i] = $root.onnx.StringStringEntryProto.fromObject(object.metadataProps[i]);
                    }
                }
                return message;
            };
    
            /**
             * Creates a plain object from a ModelProto message. Also converts values to other types if specified.
             * @function toObject
             * @memberof onnx.ModelProto
             * @static
             * @param {onnx.ModelProto} message ModelProto
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            ModelProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.opsetImport = [];
                    object.metadataProps = [];
                }
                if (options.defaults) {
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.irVersion = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.irVersion = options.longs === String ? "0" : 0;
                    object.producerName = "";
                    object.producerVersion = "";
                    object.domain = "";
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.modelVersion = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.modelVersion = options.longs === String ? "0" : 0;
                    object.docString = "";
                    object.graph = null;
                }
                if (message.irVersion != null && message.hasOwnProperty("irVersion"))
                    if (typeof message.irVersion === "number")
                        object.irVersion = options.longs === String ? String(message.irVersion) : message.irVersion;
                    else
                        object.irVersion = options.longs === String ? $util.Long.prototype.toString.call(message.irVersion) : options.longs === Number ? new $util.LongBits(message.irVersion.low >>> 0, message.irVersion.high >>> 0).toNumber() : message.irVersion;
                if (message.producerName != null && message.hasOwnProperty("producerName"))
                    object.producerName = message.producerName;
                if (message.producerVersion != null && message.hasOwnProperty("producerVersion"))
                    object.producerVersion = message.producerVersion;
                if (message.domain != null && message.hasOwnProperty("domain"))
                    object.domain = message.domain;
                if (message.modelVersion != null && message.hasOwnProperty("modelVersion"))
                    if (typeof message.modelVersion === "number")
                        object.modelVersion = options.longs === String ? String(message.modelVersion) : message.modelVersion;
                    else
                        object.modelVersion = options.longs === String ? $util.Long.prototype.toString.call(message.modelVersion) : options.longs === Number ? new $util.LongBits(message.modelVersion.low >>> 0, message.modelVersion.high >>> 0).toNumber() : message.modelVersion;
                if (message.docString != null && message.hasOwnProperty("docString"))
                    object.docString = message.docString;
                if (message.graph != null && message.hasOwnProperty("graph"))
                    object.graph = $root.onnx.GraphProto.toObject(message.graph, options);
                if (message.opsetImport && message.opsetImport.length) {
                    object.opsetImport = [];
                    for (var j = 0; j < message.opsetImport.length; ++j)
                        object.opsetImport[j] = $root.onnx.OperatorSetIdProto.toObject(message.opsetImport[j], options);
                }
                if (message.metadataProps && message.metadataProps.length) {
                    object.metadataProps = [];
                    for (var j = 0; j < message.metadataProps.length; ++j)
                        object.metadataProps[j] = $root.onnx.StringStringEntryProto.toObject(message.metadataProps[j], options);
                }
                return object;
            };
    
            /**
             * Converts this ModelProto to JSON.
             * @function toJSON
             * @memberof onnx.ModelProto
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            ModelProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ModelProto;
        })();
    
        onnx.StringStringEntryProto = (function() {
    
            /**
             * Properties of a StringStringEntryProto.
             * @memberof onnx
             * @interface IStringStringEntryProto
             * @property {string|null} [key] StringStringEntryProto key
             * @property {string|null} [value] StringStringEntryProto value
             */
    
            /**
             * Constructs a new StringStringEntryProto.
             * @memberof onnx
             * @classdesc Represents a StringStringEntryProto.
             * @implements IStringStringEntryProto
             * @constructor
             * @param {onnx.IStringStringEntryProto=} [properties] Properties to set
             */
            function StringStringEntryProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * StringStringEntryProto key.
             * @member {string} key
             * @memberof onnx.StringStringEntryProto
             * @instance
             */
            StringStringEntryProto.prototype.key = "";
    
            /**
             * StringStringEntryProto value.
             * @member {string} value
             * @memberof onnx.StringStringEntryProto
             * @instance
             */
            StringStringEntryProto.prototype.value = "";
    
            /**
             * Creates a new StringStringEntryProto instance using the specified properties.
             * @function create
             * @memberof onnx.StringStringEntryProto
             * @static
             * @param {onnx.IStringStringEntryProto=} [properties] Properties to set
             * @returns {onnx.StringStringEntryProto} StringStringEntryProto instance
             */
            StringStringEntryProto.create = function create(properties) {
                return new StringStringEntryProto(properties);
            };
    
            /**
             * Encodes the specified StringStringEntryProto message. Does not implicitly {@link onnx.StringStringEntryProto.verify|verify} messages.
             * @function encode
             * @memberof onnx.StringStringEntryProto
             * @static
             * @param {onnx.IStringStringEntryProto} message StringStringEntryProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            StringStringEntryProto.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.key != null && message.hasOwnProperty("key"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.key);
                if (message.value != null && message.hasOwnProperty("value"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.value);
                return writer;
            };
    
            /**
             * Encodes the specified StringStringEntryProto message, length delimited. Does not implicitly {@link onnx.StringStringEntryProto.verify|verify} messages.
             * @function encodeDelimited
             * @memberof onnx.StringStringEntryProto
             * @static
             * @param {onnx.IStringStringEntryProto} message StringStringEntryProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            StringStringEntryProto.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a StringStringEntryProto message from the specified reader or buffer.
             * @function decode
             * @memberof onnx.StringStringEntryProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {onnx.StringStringEntryProto} StringStringEntryProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a StringStringEntryProto message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof onnx.StringStringEntryProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {onnx.StringStringEntryProto} StringStringEntryProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            StringStringEntryProto.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a StringStringEntryProto message.
             * @function verify
             * @memberof onnx.StringStringEntryProto
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            StringStringEntryProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.key != null && message.hasOwnProperty("key"))
                    if (!$util.isString(message.key))
                        return "key: string expected";
                if (message.value != null && message.hasOwnProperty("value"))
                    if (!$util.isString(message.value))
                        return "value: string expected";
                return null;
            };
    
            /**
             * Creates a StringStringEntryProto message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof onnx.StringStringEntryProto
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {onnx.StringStringEntryProto} StringStringEntryProto
             */
            StringStringEntryProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.StringStringEntryProto)
                    return object;
                var message = new $root.onnx.StringStringEntryProto();
                if (object.key != null)
                    message.key = String(object.key);
                if (object.value != null)
                    message.value = String(object.value);
                return message;
            };
    
            /**
             * Creates a plain object from a StringStringEntryProto message. Also converts values to other types if specified.
             * @function toObject
             * @memberof onnx.StringStringEntryProto
             * @static
             * @param {onnx.StringStringEntryProto} message StringStringEntryProto
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            StringStringEntryProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.key = "";
                    object.value = "";
                }
                if (message.key != null && message.hasOwnProperty("key"))
                    object.key = message.key;
                if (message.value != null && message.hasOwnProperty("value"))
                    object.value = message.value;
                return object;
            };
    
            /**
             * Converts this StringStringEntryProto to JSON.
             * @function toJSON
             * @memberof onnx.StringStringEntryProto
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            StringStringEntryProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return StringStringEntryProto;
        })();
    
        onnx.GraphProto = (function() {
    
            /**
             * Properties of a GraphProto.
             * @memberof onnx
             * @interface IGraphProto
             * @property {Array.<onnx.INodeProto>|null} [node] GraphProto node
             * @property {string|null} [name] GraphProto name
             * @property {Array.<onnx.ITensorProto>|null} [initializer] GraphProto initializer
             * @property {string|null} [docString] GraphProto docString
             * @property {Array.<onnx.IValueInfoProto>|null} [input] GraphProto input
             * @property {Array.<onnx.IValueInfoProto>|null} [output] GraphProto output
             * @property {Array.<onnx.IValueInfoProto>|null} [valueInfo] GraphProto valueInfo
             */
    
            /**
             * Constructs a new GraphProto.
             * @memberof onnx
             * @classdesc Represents a GraphProto.
             * @implements IGraphProto
             * @constructor
             * @param {onnx.IGraphProto=} [properties] Properties to set
             */
            function GraphProto(properties) {
                this.node = [];
                this.initializer = [];
                this.input = [];
                this.output = [];
                this.valueInfo = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * GraphProto node.
             * @member {Array.<onnx.INodeProto>} node
             * @memberof onnx.GraphProto
             * @instance
             */
            GraphProto.prototype.node = $util.emptyArray;
    
            /**
             * GraphProto name.
             * @member {string} name
             * @memberof onnx.GraphProto
             * @instance
             */
            GraphProto.prototype.name = "";
    
            /**
             * GraphProto initializer.
             * @member {Array.<onnx.ITensorProto>} initializer
             * @memberof onnx.GraphProto
             * @instance
             */
            GraphProto.prototype.initializer = $util.emptyArray;
    
            /**
             * GraphProto docString.
             * @member {string} docString
             * @memberof onnx.GraphProto
             * @instance
             */
            GraphProto.prototype.docString = "";
    
            /**
             * GraphProto input.
             * @member {Array.<onnx.IValueInfoProto>} input
             * @memberof onnx.GraphProto
             * @instance
             */
            GraphProto.prototype.input = $util.emptyArray;
    
            /**
             * GraphProto output.
             * @member {Array.<onnx.IValueInfoProto>} output
             * @memberof onnx.GraphProto
             * @instance
             */
            GraphProto.prototype.output = $util.emptyArray;
    
            /**
             * GraphProto valueInfo.
             * @member {Array.<onnx.IValueInfoProto>} valueInfo
             * @memberof onnx.GraphProto
             * @instance
             */
            GraphProto.prototype.valueInfo = $util.emptyArray;
    
            /**
             * Creates a new GraphProto instance using the specified properties.
             * @function create
             * @memberof onnx.GraphProto
             * @static
             * @param {onnx.IGraphProto=} [properties] Properties to set
             * @returns {onnx.GraphProto} GraphProto instance
             */
            GraphProto.create = function create(properties) {
                return new GraphProto(properties);
            };
    
            /**
             * Encodes the specified GraphProto message. Does not implicitly {@link onnx.GraphProto.verify|verify} messages.
             * @function encode
             * @memberof onnx.GraphProto
             * @static
             * @param {onnx.IGraphProto} message GraphProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            GraphProto.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.node != null && message.node.length)
                    for (var i = 0; i < message.node.length; ++i)
                        $root.onnx.NodeProto.encode(message.node[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                if (message.name != null && message.hasOwnProperty("name"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.name);
                if (message.initializer != null && message.initializer.length)
                    for (var i = 0; i < message.initializer.length; ++i)
                        $root.onnx.TensorProto.encode(message.initializer[i], writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
                if (message.docString != null && message.hasOwnProperty("docString"))
                    writer.uint32(/* id 10, wireType 2 =*/82).string(message.docString);
                if (message.input != null && message.input.length)
                    for (var i = 0; i < message.input.length; ++i)
                        $root.onnx.ValueInfoProto.encode(message.input[i], writer.uint32(/* id 11, wireType 2 =*/90).fork()).ldelim();
                if (message.output != null && message.output.length)
                    for (var i = 0; i < message.output.length; ++i)
                        $root.onnx.ValueInfoProto.encode(message.output[i], writer.uint32(/* id 12, wireType 2 =*/98).fork()).ldelim();
                if (message.valueInfo != null && message.valueInfo.length)
                    for (var i = 0; i < message.valueInfo.length; ++i)
                        $root.onnx.ValueInfoProto.encode(message.valueInfo[i], writer.uint32(/* id 13, wireType 2 =*/106).fork()).ldelim();
                return writer;
            };
    
            /**
             * Encodes the specified GraphProto message, length delimited. Does not implicitly {@link onnx.GraphProto.verify|verify} messages.
             * @function encodeDelimited
             * @memberof onnx.GraphProto
             * @static
             * @param {onnx.IGraphProto} message GraphProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            GraphProto.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a GraphProto message from the specified reader or buffer.
             * @function decode
             * @memberof onnx.GraphProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {onnx.GraphProto} GraphProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
                    case 10:
                        message.docString = reader.string();
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
                        if (!(message.valueInfo && message.valueInfo.length))
                            message.valueInfo = [];
                        message.valueInfo.push($root.onnx.ValueInfoProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            /**
             * Decodes a GraphProto message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof onnx.GraphProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {onnx.GraphProto} GraphProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            GraphProto.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a GraphProto message.
             * @function verify
             * @memberof onnx.GraphProto
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            GraphProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.node != null && message.hasOwnProperty("node")) {
                    if (!Array.isArray(message.node))
                        return "node: array expected";
                    for (var i = 0; i < message.node.length; ++i) {
                        var error = $root.onnx.NodeProto.verify(message.node[i]);
                        if (error)
                            return "node." + error;
                    }
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.initializer != null && message.hasOwnProperty("initializer")) {
                    if (!Array.isArray(message.initializer))
                        return "initializer: array expected";
                    for (var i = 0; i < message.initializer.length; ++i) {
                        var error = $root.onnx.TensorProto.verify(message.initializer[i]);
                        if (error)
                            return "initializer." + error;
                    }
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    if (!$util.isString(message.docString))
                        return "docString: string expected";
                if (message.input != null && message.hasOwnProperty("input")) {
                    if (!Array.isArray(message.input))
                        return "input: array expected";
                    for (var i = 0; i < message.input.length; ++i) {
                        var error = $root.onnx.ValueInfoProto.verify(message.input[i]);
                        if (error)
                            return "input." + error;
                    }
                }
                if (message.output != null && message.hasOwnProperty("output")) {
                    if (!Array.isArray(message.output))
                        return "output: array expected";
                    for (var i = 0; i < message.output.length; ++i) {
                        var error = $root.onnx.ValueInfoProto.verify(message.output[i]);
                        if (error)
                            return "output." + error;
                    }
                }
                if (message.valueInfo != null && message.hasOwnProperty("valueInfo")) {
                    if (!Array.isArray(message.valueInfo))
                        return "valueInfo: array expected";
                    for (var i = 0; i < message.valueInfo.length; ++i) {
                        var error = $root.onnx.ValueInfoProto.verify(message.valueInfo[i]);
                        if (error)
                            return "valueInfo." + error;
                    }
                }
                return null;
            };
    
            /**
             * Creates a GraphProto message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof onnx.GraphProto
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {onnx.GraphProto} GraphProto
             */
            GraphProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.GraphProto)
                    return object;
                var message = new $root.onnx.GraphProto();
                if (object.node) {
                    if (!Array.isArray(object.node))
                        throw TypeError(".onnx.GraphProto.node: array expected");
                    message.node = [];
                    for (var i = 0; i < object.node.length; ++i) {
                        if (typeof object.node[i] !== "object")
                            throw TypeError(".onnx.GraphProto.node: object expected");
                        message.node[i] = $root.onnx.NodeProto.fromObject(object.node[i]);
                    }
                }
                if (object.name != null)
                    message.name = String(object.name);
                if (object.initializer) {
                    if (!Array.isArray(object.initializer))
                        throw TypeError(".onnx.GraphProto.initializer: array expected");
                    message.initializer = [];
                    for (var i = 0; i < object.initializer.length; ++i) {
                        if (typeof object.initializer[i] !== "object")
                            throw TypeError(".onnx.GraphProto.initializer: object expected");
                        message.initializer[i] = $root.onnx.TensorProto.fromObject(object.initializer[i]);
                    }
                }
                if (object.docString != null)
                    message.docString = String(object.docString);
                if (object.input) {
                    if (!Array.isArray(object.input))
                        throw TypeError(".onnx.GraphProto.input: array expected");
                    message.input = [];
                    for (var i = 0; i < object.input.length; ++i) {
                        if (typeof object.input[i] !== "object")
                            throw TypeError(".onnx.GraphProto.input: object expected");
                        message.input[i] = $root.onnx.ValueInfoProto.fromObject(object.input[i]);
                    }
                }
                if (object.output) {
                    if (!Array.isArray(object.output))
                        throw TypeError(".onnx.GraphProto.output: array expected");
                    message.output = [];
                    for (var i = 0; i < object.output.length; ++i) {
                        if (typeof object.output[i] !== "object")
                            throw TypeError(".onnx.GraphProto.output: object expected");
                        message.output[i] = $root.onnx.ValueInfoProto.fromObject(object.output[i]);
                    }
                }
                if (object.valueInfo) {
                    if (!Array.isArray(object.valueInfo))
                        throw TypeError(".onnx.GraphProto.valueInfo: array expected");
                    message.valueInfo = [];
                    for (var i = 0; i < object.valueInfo.length; ++i) {
                        if (typeof object.valueInfo[i] !== "object")
                            throw TypeError(".onnx.GraphProto.valueInfo: object expected");
                        message.valueInfo[i] = $root.onnx.ValueInfoProto.fromObject(object.valueInfo[i]);
                    }
                }
                return message;
            };
    
            /**
             * Creates a plain object from a GraphProto message. Also converts values to other types if specified.
             * @function toObject
             * @memberof onnx.GraphProto
             * @static
             * @param {onnx.GraphProto} message GraphProto
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            GraphProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.node = [];
                    object.initializer = [];
                    object.input = [];
                    object.output = [];
                    object.valueInfo = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.docString = "";
                }
                if (message.node && message.node.length) {
                    object.node = [];
                    for (var j = 0; j < message.node.length; ++j)
                        object.node[j] = $root.onnx.NodeProto.toObject(message.node[j], options);
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.initializer && message.initializer.length) {
                    object.initializer = [];
                    for (var j = 0; j < message.initializer.length; ++j)
                        object.initializer[j] = $root.onnx.TensorProto.toObject(message.initializer[j], options);
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    object.docString = message.docString;
                if (message.input && message.input.length) {
                    object.input = [];
                    for (var j = 0; j < message.input.length; ++j)
                        object.input[j] = $root.onnx.ValueInfoProto.toObject(message.input[j], options);
                }
                if (message.output && message.output.length) {
                    object.output = [];
                    for (var j = 0; j < message.output.length; ++j)
                        object.output[j] = $root.onnx.ValueInfoProto.toObject(message.output[j], options);
                }
                if (message.valueInfo && message.valueInfo.length) {
                    object.valueInfo = [];
                    for (var j = 0; j < message.valueInfo.length; ++j)
                        object.valueInfo[j] = $root.onnx.ValueInfoProto.toObject(message.valueInfo[j], options);
                }
                return object;
            };
    
            /**
             * Converts this GraphProto to JSON.
             * @function toJSON
             * @memberof onnx.GraphProto
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            GraphProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return GraphProto;
        })();
    
        onnx.TensorProto = (function() {
    
            /**
             * Properties of a TensorProto.
             * @memberof onnx
             * @interface ITensorProto
             * @property {Array.<number|Long>|null} [dims] TensorProto dims
             * @property {onnx.TensorProto.DataType|null} [dataType] TensorProto dataType
             * @property {onnx.TensorProto.ISegment|null} [segment] TensorProto segment
             * @property {Array.<number>|null} [floatData] TensorProto floatData
             * @property {Array.<number>|null} [int32Data] TensorProto int32Data
             * @property {Array.<Uint8Array>|null} [stringData] TensorProto stringData
             * @property {Array.<number|Long>|null} [int64Data] TensorProto int64Data
             * @property {string|null} [name] TensorProto name
             * @property {string|null} [docString] TensorProto docString
             * @property {Uint8Array|null} [rawData] TensorProto rawData
             * @property {Array.<number>|null} [doubleData] TensorProto doubleData
             * @property {Array.<number|Long>|null} [uint64Data] TensorProto uint64Data
             */
    
            /**
             * Constructs a new TensorProto.
             * @memberof onnx
             * @classdesc Represents a TensorProto.
             * @implements ITensorProto
             * @constructor
             * @param {onnx.ITensorProto=} [properties] Properties to set
             */
            function TensorProto(properties) {
                this.dims = [];
                this.floatData = [];
                this.int32Data = [];
                this.stringData = [];
                this.int64Data = [];
                this.doubleData = [];
                this.uint64Data = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * TensorProto dims.
             * @member {Array.<number|Long>} dims
             * @memberof onnx.TensorProto
             * @instance
             */
            TensorProto.prototype.dims = $util.emptyArray;
    
            /**
             * TensorProto dataType.
             * @member {onnx.TensorProto.DataType} dataType
             * @memberof onnx.TensorProto
             * @instance
             */
            TensorProto.prototype.dataType = 0;
    
            /**
             * TensorProto segment.
             * @member {onnx.TensorProto.ISegment|null|undefined} segment
             * @memberof onnx.TensorProto
             * @instance
             */
            TensorProto.prototype.segment = null;
    
            /**
             * TensorProto floatData.
             * @member {Array.<number>} floatData
             * @memberof onnx.TensorProto
             * @instance
             */
            TensorProto.prototype.floatData = $util.emptyArray;
    
            /**
             * TensorProto int32Data.
             * @member {Array.<number>} int32Data
             * @memberof onnx.TensorProto
             * @instance
             */
            TensorProto.prototype.int32Data = $util.emptyArray;
    
            /**
             * TensorProto stringData.
             * @member {Array.<Uint8Array>} stringData
             * @memberof onnx.TensorProto
             * @instance
             */
            TensorProto.prototype.stringData = $util.emptyArray;
    
            /**
             * TensorProto int64Data.
             * @member {Array.<number|Long>} int64Data
             * @memberof onnx.TensorProto
             * @instance
             */
            TensorProto.prototype.int64Data = $util.emptyArray;
    
            /**
             * TensorProto name.
             * @member {string} name
             * @memberof onnx.TensorProto
             * @instance
             */
            TensorProto.prototype.name = "";
    
            /**
             * TensorProto docString.
             * @member {string} docString
             * @memberof onnx.TensorProto
             * @instance
             */
            TensorProto.prototype.docString = "";
    
            /**
             * TensorProto rawData.
             * @member {Uint8Array} rawData
             * @memberof onnx.TensorProto
             * @instance
             */
            TensorProto.prototype.rawData = $util.newBuffer([]);
    
            /**
             * TensorProto doubleData.
             * @member {Array.<number>} doubleData
             * @memberof onnx.TensorProto
             * @instance
             */
            TensorProto.prototype.doubleData = $util.emptyArray;
    
            /**
             * TensorProto uint64Data.
             * @member {Array.<number|Long>} uint64Data
             * @memberof onnx.TensorProto
             * @instance
             */
            TensorProto.prototype.uint64Data = $util.emptyArray;
    
            /**
             * Creates a new TensorProto instance using the specified properties.
             * @function create
             * @memberof onnx.TensorProto
             * @static
             * @param {onnx.ITensorProto=} [properties] Properties to set
             * @returns {onnx.TensorProto} TensorProto instance
             */
            TensorProto.create = function create(properties) {
                return new TensorProto(properties);
            };
    
            /**
             * Encodes the specified TensorProto message. Does not implicitly {@link onnx.TensorProto.verify|verify} messages.
             * @function encode
             * @memberof onnx.TensorProto
             * @static
             * @param {onnx.ITensorProto} message TensorProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TensorProto.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.dims != null && message.dims.length)
                    for (var i = 0; i < message.dims.length; ++i)
                        writer.uint32(/* id 1, wireType 0 =*/8).int64(message.dims[i]);
                if (message.dataType != null && message.hasOwnProperty("dataType"))
                    writer.uint32(/* id 2, wireType 0 =*/16).int32(message.dataType);
                if (message.segment != null && message.hasOwnProperty("segment"))
                    $root.onnx.TensorProto.Segment.encode(message.segment, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
                if (message.floatData != null && message.floatData.length) {
                    writer.uint32(/* id 4, wireType 2 =*/34).fork();
                    for (var i = 0; i < message.floatData.length; ++i)
                        writer.float(message.floatData[i]);
                    writer.ldelim();
                }
                if (message.int32Data != null && message.int32Data.length) {
                    writer.uint32(/* id 5, wireType 2 =*/42).fork();
                    for (var i = 0; i < message.int32Data.length; ++i)
                        writer.int32(message.int32Data[i]);
                    writer.ldelim();
                }
                if (message.stringData != null && message.stringData.length)
                    for (var i = 0; i < message.stringData.length; ++i)
                        writer.uint32(/* id 6, wireType 2 =*/50).bytes(message.stringData[i]);
                if (message.int64Data != null && message.int64Data.length) {
                    writer.uint32(/* id 7, wireType 2 =*/58).fork();
                    for (var i = 0; i < message.int64Data.length; ++i)
                        writer.int64(message.int64Data[i]);
                    writer.ldelim();
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    writer.uint32(/* id 8, wireType 2 =*/66).string(message.name);
                if (message.rawData != null && message.hasOwnProperty("rawData"))
                    writer.uint32(/* id 9, wireType 2 =*/74).bytes(message.rawData);
                if (message.doubleData != null && message.doubleData.length) {
                    writer.uint32(/* id 10, wireType 2 =*/82).fork();
                    for (var i = 0; i < message.doubleData.length; ++i)
                        writer.double(message.doubleData[i]);
                    writer.ldelim();
                }
                if (message.uint64Data != null && message.uint64Data.length) {
                    writer.uint32(/* id 11, wireType 2 =*/90).fork();
                    for (var i = 0; i < message.uint64Data.length; ++i)
                        writer.uint64(message.uint64Data[i]);
                    writer.ldelim();
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    writer.uint32(/* id 12, wireType 2 =*/98).string(message.docString);
                return writer;
            };
    
            /**
             * Encodes the specified TensorProto message, length delimited. Does not implicitly {@link onnx.TensorProto.verify|verify} messages.
             * @function encodeDelimited
             * @memberof onnx.TensorProto
             * @static
             * @param {onnx.ITensorProto} message TensorProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TensorProto.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a TensorProto message from the specified reader or buffer.
             * @function decode
             * @memberof onnx.TensorProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {onnx.TensorProto} TensorProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
                        message.dataType = reader.int32();
                        break;
                    case 3:
                        message.segment = $root.onnx.TensorProto.Segment.decode(reader, reader.uint32());
                        break;
                    case 4:
                        if (!(message.floatData && message.floatData.length))
                            message.floatData = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.floatData.push(reader.float());
                        } else
                            message.floatData.push(reader.float());
                        break;
                    case 5:
                        if (!(message.int32Data && message.int32Data.length))
                            message.int32Data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.int32Data.push(reader.int32());
                        } else
                            message.int32Data.push(reader.int32());
                        break;
                    case 6:
                        if (!(message.stringData && message.stringData.length))
                            message.stringData = [];
                        message.stringData.push(reader.bytes());
                        break;
                    case 7:
                        if (!(message.int64Data && message.int64Data.length))
                            message.int64Data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.int64Data.push(reader.int64());
                        } else
                            message.int64Data.push(reader.int64());
                        break;
                    case 8:
                        message.name = reader.string();
                        break;
                    case 12:
                        message.docString = reader.string();
                        break;
                    case 9:
                        message.rawData = reader.bytes();
                        break;
                    case 10:
                        if (!(message.doubleData && message.doubleData.length))
                            message.doubleData = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.doubleData.push(reader.double());
                        } else
                            message.doubleData.push(reader.double());
                        break;
                    case 11:
                        if (!(message.uint64Data && message.uint64Data.length))
                            message.uint64Data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.uint64Data.push(reader.uint64());
                        } else
                            message.uint64Data.push(reader.uint64());
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
             * @memberof onnx.TensorProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {onnx.TensorProto} TensorProto
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
             * @memberof onnx.TensorProto
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            TensorProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.dims != null && message.hasOwnProperty("dims")) {
                    if (!Array.isArray(message.dims))
                        return "dims: array expected";
                    for (var i = 0; i < message.dims.length; ++i)
                        if (!$util.isInteger(message.dims[i]) && !(message.dims[i] && $util.isInteger(message.dims[i].low) && $util.isInteger(message.dims[i].high)))
                            return "dims: integer|Long[] expected";
                }
                if (message.dataType != null && message.hasOwnProperty("dataType"))
                    switch (message.dataType) {
                    default:
                        return "dataType: enum value expected";
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
                        break;
                    }
                if (message.segment != null && message.hasOwnProperty("segment")) {
                    var error = $root.onnx.TensorProto.Segment.verify(message.segment);
                    if (error)
                        return "segment." + error;
                }
                if (message.floatData != null && message.hasOwnProperty("floatData")) {
                    if (!Array.isArray(message.floatData))
                        return "floatData: array expected";
                    for (var i = 0; i < message.floatData.length; ++i)
                        if (typeof message.floatData[i] !== "number")
                            return "floatData: number[] expected";
                }
                if (message.int32Data != null && message.hasOwnProperty("int32Data")) {
                    if (!Array.isArray(message.int32Data))
                        return "int32Data: array expected";
                    for (var i = 0; i < message.int32Data.length; ++i)
                        if (!$util.isInteger(message.int32Data[i]))
                            return "int32Data: integer[] expected";
                }
                if (message.stringData != null && message.hasOwnProperty("stringData")) {
                    if (!Array.isArray(message.stringData))
                        return "stringData: array expected";
                    for (var i = 0; i < message.stringData.length; ++i)
                        if (!(message.stringData[i] && typeof message.stringData[i].length === "number" || $util.isString(message.stringData[i])))
                            return "stringData: buffer[] expected";
                }
                if (message.int64Data != null && message.hasOwnProperty("int64Data")) {
                    if (!Array.isArray(message.int64Data))
                        return "int64Data: array expected";
                    for (var i = 0; i < message.int64Data.length; ++i)
                        if (!$util.isInteger(message.int64Data[i]) && !(message.int64Data[i] && $util.isInteger(message.int64Data[i].low) && $util.isInteger(message.int64Data[i].high)))
                            return "int64Data: integer|Long[] expected";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.docString != null && message.hasOwnProperty("docString"))
                    if (!$util.isString(message.docString))
                        return "docString: string expected";
                if (message.rawData != null && message.hasOwnProperty("rawData"))
                    if (!(message.rawData && typeof message.rawData.length === "number" || $util.isString(message.rawData)))
                        return "rawData: buffer expected";
                if (message.doubleData != null && message.hasOwnProperty("doubleData")) {
                    if (!Array.isArray(message.doubleData))
                        return "doubleData: array expected";
                    for (var i = 0; i < message.doubleData.length; ++i)
                        if (typeof message.doubleData[i] !== "number")
                            return "doubleData: number[] expected";
                }
                if (message.uint64Data != null && message.hasOwnProperty("uint64Data")) {
                    if (!Array.isArray(message.uint64Data))
                        return "uint64Data: array expected";
                    for (var i = 0; i < message.uint64Data.length; ++i)
                        if (!$util.isInteger(message.uint64Data[i]) && !(message.uint64Data[i] && $util.isInteger(message.uint64Data[i].low) && $util.isInteger(message.uint64Data[i].high)))
                            return "uint64Data: integer|Long[] expected";
                }
                return null;
            };
    
            /**
             * Creates a TensorProto message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof onnx.TensorProto
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {onnx.TensorProto} TensorProto
             */
            TensorProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.TensorProto)
                    return object;
                var message = new $root.onnx.TensorProto();
                if (object.dims) {
                    if (!Array.isArray(object.dims))
                        throw TypeError(".onnx.TensorProto.dims: array expected");
                    message.dims = [];
                    for (var i = 0; i < object.dims.length; ++i)
                        if ($util.Long)
                            (message.dims[i] = $util.Long.fromValue(object.dims[i])).unsigned = false;
                        else if (typeof object.dims[i] === "string")
                            message.dims[i] = parseInt(object.dims[i], 10);
                        else if (typeof object.dims[i] === "number")
                            message.dims[i] = object.dims[i];
                        else if (typeof object.dims[i] === "object")
                            message.dims[i] = new $util.LongBits(object.dims[i].low >>> 0, object.dims[i].high >>> 0).toNumber();
                }
                switch (object.dataType) {
                case "UNDEFINED":
                case 0:
                    message.dataType = 0;
                    break;
                case "FLOAT":
                case 1:
                    message.dataType = 1;
                    break;
                case "UINT8":
                case 2:
                    message.dataType = 2;
                    break;
                case "INT8":
                case 3:
                    message.dataType = 3;
                    break;
                case "UINT16":
                case 4:
                    message.dataType = 4;
                    break;
                case "INT16":
                case 5:
                    message.dataType = 5;
                    break;
                case "INT32":
                case 6:
                    message.dataType = 6;
                    break;
                case "INT64":
                case 7:
                    message.dataType = 7;
                    break;
                case "STRING":
                case 8:
                    message.dataType = 8;
                    break;
                case "BOOL":
                case 9:
                    message.dataType = 9;
                    break;
                case "FLOAT16":
                case 10:
                    message.dataType = 10;
                    break;
                case "DOUBLE":
                case 11:
                    message.dataType = 11;
                    break;
                case "UINT32":
                case 12:
                    message.dataType = 12;
                    break;
                case "UINT64":
                case 13:
                    message.dataType = 13;
                    break;
                case "COMPLEX64":
                case 14:
                    message.dataType = 14;
                    break;
                case "COMPLEX128":
                case 15:
                    message.dataType = 15;
                    break;
                }
                if (object.segment != null) {
                    if (typeof object.segment !== "object")
                        throw TypeError(".onnx.TensorProto.segment: object expected");
                    message.segment = $root.onnx.TensorProto.Segment.fromObject(object.segment);
                }
                if (object.floatData) {
                    if (!Array.isArray(object.floatData))
                        throw TypeError(".onnx.TensorProto.floatData: array expected");
                    message.floatData = [];
                    for (var i = 0; i < object.floatData.length; ++i)
                        message.floatData[i] = Number(object.floatData[i]);
                }
                if (object.int32Data) {
                    if (!Array.isArray(object.int32Data))
                        throw TypeError(".onnx.TensorProto.int32Data: array expected");
                    message.int32Data = [];
                    for (var i = 0; i < object.int32Data.length; ++i)
                        message.int32Data[i] = object.int32Data[i] | 0;
                }
                if (object.stringData) {
                    if (!Array.isArray(object.stringData))
                        throw TypeError(".onnx.TensorProto.stringData: array expected");
                    message.stringData = [];
                    for (var i = 0; i < object.stringData.length; ++i)
                        if (typeof object.stringData[i] === "string")
                            $util.base64.decode(object.stringData[i], message.stringData[i] = $util.newBuffer($util.base64.length(object.stringData[i])), 0);
                        else if (object.stringData[i].length)
                            message.stringData[i] = object.stringData[i];
                }
                if (object.int64Data) {
                    if (!Array.isArray(object.int64Data))
                        throw TypeError(".onnx.TensorProto.int64Data: array expected");
                    message.int64Data = [];
                    for (var i = 0; i < object.int64Data.length; ++i)
                        if ($util.Long)
                            (message.int64Data[i] = $util.Long.fromValue(object.int64Data[i])).unsigned = false;
                        else if (typeof object.int64Data[i] === "string")
                            message.int64Data[i] = parseInt(object.int64Data[i], 10);
                        else if (typeof object.int64Data[i] === "number")
                            message.int64Data[i] = object.int64Data[i];
                        else if (typeof object.int64Data[i] === "object")
                            message.int64Data[i] = new $util.LongBits(object.int64Data[i].low >>> 0, object.int64Data[i].high >>> 0).toNumber();
                }
                if (object.name != null)
                    message.name = String(object.name);
                if (object.docString != null)
                    message.docString = String(object.docString);
                if (object.rawData != null)
                    if (typeof object.rawData === "string")
                        $util.base64.decode(object.rawData, message.rawData = $util.newBuffer($util.base64.length(object.rawData)), 0);
                    else if (object.rawData.length)
                        message.rawData = object.rawData;
                if (object.doubleData) {
                    if (!Array.isArray(object.doubleData))
                        throw TypeError(".onnx.TensorProto.doubleData: array expected");
                    message.doubleData = [];
                    for (var i = 0; i < object.doubleData.length; ++i)
                        message.doubleData[i] = Number(object.doubleData[i]);
                }
                if (object.uint64Data) {
                    if (!Array.isArray(object.uint64Data))
                        throw TypeError(".onnx.TensorProto.uint64Data: array expected");
                    message.uint64Data = [];
                    for (var i = 0; i < object.uint64Data.length; ++i)
                        if ($util.Long)
                            (message.uint64Data[i] = $util.Long.fromValue(object.uint64Data[i])).unsigned = true;
                        else if (typeof object.uint64Data[i] === "string")
                            message.uint64Data[i] = parseInt(object.uint64Data[i], 10);
                        else if (typeof object.uint64Data[i] === "number")
                            message.uint64Data[i] = object.uint64Data[i];
                        else if (typeof object.uint64Data[i] === "object")
                            message.uint64Data[i] = new $util.LongBits(object.uint64Data[i].low >>> 0, object.uint64Data[i].high >>> 0).toNumber(true);
                }
                return message;
            };
    
            /**
             * Creates a plain object from a TensorProto message. Also converts values to other types if specified.
             * @function toObject
             * @memberof onnx.TensorProto
             * @static
             * @param {onnx.TensorProto} message TensorProto
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            TensorProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.dims = [];
                    object.floatData = [];
                    object.int32Data = [];
                    object.stringData = [];
                    object.int64Data = [];
                    object.doubleData = [];
                    object.uint64Data = [];
                }
                if (options.defaults) {
                    object.dataType = options.enums === String ? "UNDEFINED" : 0;
                    object.segment = null;
                    object.name = "";
                    object.rawData = options.bytes === String ? "" : [];
                    object.docString = "";
                }
                if (message.dims && message.dims.length) {
                    object.dims = [];
                    for (var j = 0; j < message.dims.length; ++j)
                        if (typeof message.dims[j] === "number")
                            object.dims[j] = options.longs === String ? String(message.dims[j]) : message.dims[j];
                        else
                            object.dims[j] = options.longs === String ? $util.Long.prototype.toString.call(message.dims[j]) : options.longs === Number ? new $util.LongBits(message.dims[j].low >>> 0, message.dims[j].high >>> 0).toNumber() : message.dims[j];
                }
                if (message.dataType != null && message.hasOwnProperty("dataType"))
                    object.dataType = options.enums === String ? $root.onnx.TensorProto.DataType[message.dataType] : message.dataType;
                if (message.segment != null && message.hasOwnProperty("segment"))
                    object.segment = $root.onnx.TensorProto.Segment.toObject(message.segment, options);
                if (message.floatData && message.floatData.length) {
                    object.floatData = [];
                    for (var j = 0; j < message.floatData.length; ++j)
                        object.floatData[j] = options.json && !isFinite(message.floatData[j]) ? String(message.floatData[j]) : message.floatData[j];
                }
                if (message.int32Data && message.int32Data.length) {
                    object.int32Data = [];
                    for (var j = 0; j < message.int32Data.length; ++j)
                        object.int32Data[j] = message.int32Data[j];
                }
                if (message.stringData && message.stringData.length) {
                    object.stringData = [];
                    for (var j = 0; j < message.stringData.length; ++j)
                        object.stringData[j] = options.bytes === String ? $util.base64.encode(message.stringData[j], 0, message.stringData[j].length) : options.bytes === Array ? Array.prototype.slice.call(message.stringData[j]) : message.stringData[j];
                }
                if (message.int64Data && message.int64Data.length) {
                    object.int64Data = [];
                    for (var j = 0; j < message.int64Data.length; ++j)
                        if (typeof message.int64Data[j] === "number")
                            object.int64Data[j] = options.longs === String ? String(message.int64Data[j]) : message.int64Data[j];
                        else
                            object.int64Data[j] = options.longs === String ? $util.Long.prototype.toString.call(message.int64Data[j]) : options.longs === Number ? new $util.LongBits(message.int64Data[j].low >>> 0, message.int64Data[j].high >>> 0).toNumber() : message.int64Data[j];
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.rawData != null && message.hasOwnProperty("rawData"))
                    object.rawData = options.bytes === String ? $util.base64.encode(message.rawData, 0, message.rawData.length) : options.bytes === Array ? Array.prototype.slice.call(message.rawData) : message.rawData;
                if (message.doubleData && message.doubleData.length) {
                    object.doubleData = [];
                    for (var j = 0; j < message.doubleData.length; ++j)
                        object.doubleData[j] = options.json && !isFinite(message.doubleData[j]) ? String(message.doubleData[j]) : message.doubleData[j];
                }
                if (message.uint64Data && message.uint64Data.length) {
                    object.uint64Data = [];
                    for (var j = 0; j < message.uint64Data.length; ++j)
                        if (typeof message.uint64Data[j] === "number")
                            object.uint64Data[j] = options.longs === String ? String(message.uint64Data[j]) : message.uint64Data[j];
                        else
                            object.uint64Data[j] = options.longs === String ? $util.Long.prototype.toString.call(message.uint64Data[j]) : options.longs === Number ? new $util.LongBits(message.uint64Data[j].low >>> 0, message.uint64Data[j].high >>> 0).toNumber(true) : message.uint64Data[j];
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    object.docString = message.docString;
                return object;
            };
    
            /**
             * Converts this TensorProto to JSON.
             * @function toJSON
             * @memberof onnx.TensorProto
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            TensorProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            /**
             * DataType enum.
             * @name onnx.TensorProto.DataType
             * @enum {string}
             * @property {number} UNDEFINED=0 UNDEFINED value
             * @property {number} FLOAT=1 FLOAT value
             * @property {number} UINT8=2 UINT8 value
             * @property {number} INT8=3 INT8 value
             * @property {number} UINT16=4 UINT16 value
             * @property {number} INT16=5 INT16 value
             * @property {number} INT32=6 INT32 value
             * @property {number} INT64=7 INT64 value
             * @property {number} STRING=8 STRING value
             * @property {number} BOOL=9 BOOL value
             * @property {number} FLOAT16=10 FLOAT16 value
             * @property {number} DOUBLE=11 DOUBLE value
             * @property {number} UINT32=12 UINT32 value
             * @property {number} UINT64=13 UINT64 value
             * @property {number} COMPLEX64=14 COMPLEX64 value
             * @property {number} COMPLEX128=15 COMPLEX128 value
             */
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
                return values;
            })();
    
            TensorProto.Segment = (function() {
    
                /**
                 * Properties of a Segment.
                 * @memberof onnx.TensorProto
                 * @interface ISegment
                 * @property {number|Long|null} [begin] Segment begin
                 * @property {number|Long|null} [end] Segment end
                 */
    
                /**
                 * Constructs a new Segment.
                 * @memberof onnx.TensorProto
                 * @classdesc Represents a Segment.
                 * @implements ISegment
                 * @constructor
                 * @param {onnx.TensorProto.ISegment=} [properties] Properties to set
                 */
                function Segment(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * Segment begin.
                 * @member {number|Long} begin
                 * @memberof onnx.TensorProto.Segment
                 * @instance
                 */
                Segment.prototype.begin = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                /**
                 * Segment end.
                 * @member {number|Long} end
                 * @memberof onnx.TensorProto.Segment
                 * @instance
                 */
                Segment.prototype.end = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                /**
                 * Creates a new Segment instance using the specified properties.
                 * @function create
                 * @memberof onnx.TensorProto.Segment
                 * @static
                 * @param {onnx.TensorProto.ISegment=} [properties] Properties to set
                 * @returns {onnx.TensorProto.Segment} Segment instance
                 */
                Segment.create = function create(properties) {
                    return new Segment(properties);
                };
    
                /**
                 * Encodes the specified Segment message. Does not implicitly {@link onnx.TensorProto.Segment.verify|verify} messages.
                 * @function encode
                 * @memberof onnx.TensorProto.Segment
                 * @static
                 * @param {onnx.TensorProto.ISegment} message Segment message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Segment.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.begin != null && message.hasOwnProperty("begin"))
                        writer.uint32(/* id 1, wireType 0 =*/8).int64(message.begin);
                    if (message.end != null && message.hasOwnProperty("end"))
                        writer.uint32(/* id 2, wireType 0 =*/16).int64(message.end);
                    return writer;
                };
    
                /**
                 * Encodes the specified Segment message, length delimited. Does not implicitly {@link onnx.TensorProto.Segment.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof onnx.TensorProto.Segment
                 * @static
                 * @param {onnx.TensorProto.ISegment} message Segment message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Segment.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes a Segment message from the specified reader or buffer.
                 * @function decode
                 * @memberof onnx.TensorProto.Segment
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {onnx.TensorProto.Segment} Segment
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
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
    
                /**
                 * Decodes a Segment message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof onnx.TensorProto.Segment
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {onnx.TensorProto.Segment} Segment
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                Segment.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies a Segment message.
                 * @function verify
                 * @memberof onnx.TensorProto.Segment
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
                Segment.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.begin != null && message.hasOwnProperty("begin"))
                        if (!$util.isInteger(message.begin) && !(message.begin && $util.isInteger(message.begin.low) && $util.isInteger(message.begin.high)))
                            return "begin: integer|Long expected";
                    if (message.end != null && message.hasOwnProperty("end"))
                        if (!$util.isInteger(message.end) && !(message.end && $util.isInteger(message.end.low) && $util.isInteger(message.end.high)))
                            return "end: integer|Long expected";
                    return null;
                };
    
                /**
                 * Creates a Segment message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof onnx.TensorProto.Segment
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {onnx.TensorProto.Segment} Segment
                 */
                Segment.fromObject = function fromObject(object) {
                    if (object instanceof $root.onnx.TensorProto.Segment)
                        return object;
                    var message = new $root.onnx.TensorProto.Segment();
                    if (object.begin != null)
                        if ($util.Long)
                            (message.begin = $util.Long.fromValue(object.begin)).unsigned = false;
                        else if (typeof object.begin === "string")
                            message.begin = parseInt(object.begin, 10);
                        else if (typeof object.begin === "number")
                            message.begin = object.begin;
                        else if (typeof object.begin === "object")
                            message.begin = new $util.LongBits(object.begin.low >>> 0, object.begin.high >>> 0).toNumber();
                    if (object.end != null)
                        if ($util.Long)
                            (message.end = $util.Long.fromValue(object.end)).unsigned = false;
                        else if (typeof object.end === "string")
                            message.end = parseInt(object.end, 10);
                        else if (typeof object.end === "number")
                            message.end = object.end;
                        else if (typeof object.end === "object")
                            message.end = new $util.LongBits(object.end.low >>> 0, object.end.high >>> 0).toNumber();
                    return message;
                };
    
                /**
                 * Creates a plain object from a Segment message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof onnx.TensorProto.Segment
                 * @static
                 * @param {onnx.TensorProto.Segment} message Segment
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
                Segment.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        if ($util.Long) {
                            var long = new $util.Long(0, 0, false);
                            object.begin = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                        } else
                            object.begin = options.longs === String ? "0" : 0;
                        if ($util.Long) {
                            var long = new $util.Long(0, 0, false);
                            object.end = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                        } else
                            object.end = options.longs === String ? "0" : 0;
                    }
                    if (message.begin != null && message.hasOwnProperty("begin"))
                        if (typeof message.begin === "number")
                            object.begin = options.longs === String ? String(message.begin) : message.begin;
                        else
                            object.begin = options.longs === String ? $util.Long.prototype.toString.call(message.begin) : options.longs === Number ? new $util.LongBits(message.begin.low >>> 0, message.begin.high >>> 0).toNumber() : message.begin;
                    if (message.end != null && message.hasOwnProperty("end"))
                        if (typeof message.end === "number")
                            object.end = options.longs === String ? String(message.end) : message.end;
                        else
                            object.end = options.longs === String ? $util.Long.prototype.toString.call(message.end) : options.longs === Number ? new $util.LongBits(message.end.low >>> 0, message.end.high >>> 0).toNumber() : message.end;
                    return object;
                };
    
                /**
                 * Converts this Segment to JSON.
                 * @function toJSON
                 * @memberof onnx.TensorProto.Segment
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                Segment.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return Segment;
            })();
    
            return TensorProto;
        })();
    
        onnx.TensorShapeProto = (function() {
    
            /**
             * Properties of a TensorShapeProto.
             * @memberof onnx
             * @interface ITensorShapeProto
             * @property {Array.<onnx.TensorShapeProto.IDimension>|null} [dim] TensorShapeProto dim
             */
    
            /**
             * Constructs a new TensorShapeProto.
             * @memberof onnx
             * @classdesc Represents a TensorShapeProto.
             * @implements ITensorShapeProto
             * @constructor
             * @param {onnx.ITensorShapeProto=} [properties] Properties to set
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
             * @member {Array.<onnx.TensorShapeProto.IDimension>} dim
             * @memberof onnx.TensorShapeProto
             * @instance
             */
            TensorShapeProto.prototype.dim = $util.emptyArray;
    
            /**
             * Creates a new TensorShapeProto instance using the specified properties.
             * @function create
             * @memberof onnx.TensorShapeProto
             * @static
             * @param {onnx.ITensorShapeProto=} [properties] Properties to set
             * @returns {onnx.TensorShapeProto} TensorShapeProto instance
             */
            TensorShapeProto.create = function create(properties) {
                return new TensorShapeProto(properties);
            };
    
            /**
             * Encodes the specified TensorShapeProto message. Does not implicitly {@link onnx.TensorShapeProto.verify|verify} messages.
             * @function encode
             * @memberof onnx.TensorShapeProto
             * @static
             * @param {onnx.ITensorShapeProto} message TensorShapeProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TensorShapeProto.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.dim != null && message.dim.length)
                    for (var i = 0; i < message.dim.length; ++i)
                        $root.onnx.TensorShapeProto.Dimension.encode(message.dim[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                return writer;
            };
    
            /**
             * Encodes the specified TensorShapeProto message, length delimited. Does not implicitly {@link onnx.TensorShapeProto.verify|verify} messages.
             * @function encodeDelimited
             * @memberof onnx.TensorShapeProto
             * @static
             * @param {onnx.ITensorShapeProto} message TensorShapeProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TensorShapeProto.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a TensorShapeProto message from the specified reader or buffer.
             * @function decode
             * @memberof onnx.TensorShapeProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {onnx.TensorShapeProto} TensorShapeProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes a TensorShapeProto message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof onnx.TensorShapeProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {onnx.TensorShapeProto} TensorShapeProto
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
             * @memberof onnx.TensorShapeProto
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
                        var error = $root.onnx.TensorShapeProto.Dimension.verify(message.dim[i]);
                        if (error)
                            return "dim." + error;
                    }
                }
                return null;
            };
    
            /**
             * Creates a TensorShapeProto message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof onnx.TensorShapeProto
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {onnx.TensorShapeProto} TensorShapeProto
             */
            TensorShapeProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.TensorShapeProto)
                    return object;
                var message = new $root.onnx.TensorShapeProto();
                if (object.dim) {
                    if (!Array.isArray(object.dim))
                        throw TypeError(".onnx.TensorShapeProto.dim: array expected");
                    message.dim = [];
                    for (var i = 0; i < object.dim.length; ++i) {
                        if (typeof object.dim[i] !== "object")
                            throw TypeError(".onnx.TensorShapeProto.dim: object expected");
                        message.dim[i] = $root.onnx.TensorShapeProto.Dimension.fromObject(object.dim[i]);
                    }
                }
                return message;
            };
    
            /**
             * Creates a plain object from a TensorShapeProto message. Also converts values to other types if specified.
             * @function toObject
             * @memberof onnx.TensorShapeProto
             * @static
             * @param {onnx.TensorShapeProto} message TensorShapeProto
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            TensorShapeProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.dim = [];
                if (message.dim && message.dim.length) {
                    object.dim = [];
                    for (var j = 0; j < message.dim.length; ++j)
                        object.dim[j] = $root.onnx.TensorShapeProto.Dimension.toObject(message.dim[j], options);
                }
                return object;
            };
    
            /**
             * Converts this TensorShapeProto to JSON.
             * @function toJSON
             * @memberof onnx.TensorShapeProto
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            TensorShapeProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            TensorShapeProto.Dimension = (function() {
    
                /**
                 * Properties of a Dimension.
                 * @memberof onnx.TensorShapeProto
                 * @interface IDimension
                 * @property {number|Long|null} [dimValue] Dimension dimValue
                 * @property {string|null} [dimParam] Dimension dimParam
                 */
    
                /**
                 * Constructs a new Dimension.
                 * @memberof onnx.TensorShapeProto
                 * @classdesc Represents a Dimension.
                 * @implements IDimension
                 * @constructor
                 * @param {onnx.TensorShapeProto.IDimension=} [properties] Properties to set
                 */
                function Dimension(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * Dimension dimValue.
                 * @member {number|Long} dimValue
                 * @memberof onnx.TensorShapeProto.Dimension
                 * @instance
                 */
                Dimension.prototype.dimValue = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                /**
                 * Dimension dimParam.
                 * @member {string} dimParam
                 * @memberof onnx.TensorShapeProto.Dimension
                 * @instance
                 */
                Dimension.prototype.dimParam = "";
    
                // OneOf field names bound to virtual getters and setters
                var $oneOfFields;
    
                /**
                 * Dimension value.
                 * @member {"dimValue"|"dimParam"|undefined} value
                 * @memberof onnx.TensorShapeProto.Dimension
                 * @instance
                 */
                Object.defineProperty(Dimension.prototype, "value", {
                    get: $util.oneOfGetter($oneOfFields = ["dimValue", "dimParam"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                /**
                 * Creates a new Dimension instance using the specified properties.
                 * @function create
                 * @memberof onnx.TensorShapeProto.Dimension
                 * @static
                 * @param {onnx.TensorShapeProto.IDimension=} [properties] Properties to set
                 * @returns {onnx.TensorShapeProto.Dimension} Dimension instance
                 */
                Dimension.create = function create(properties) {
                    return new Dimension(properties);
                };
    
                /**
                 * Encodes the specified Dimension message. Does not implicitly {@link onnx.TensorShapeProto.Dimension.verify|verify} messages.
                 * @function encode
                 * @memberof onnx.TensorShapeProto.Dimension
                 * @static
                 * @param {onnx.TensorShapeProto.IDimension} message Dimension message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Dimension.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.dimValue != null && message.hasOwnProperty("dimValue"))
                        writer.uint32(/* id 1, wireType 0 =*/8).int64(message.dimValue);
                    if (message.dimParam != null && message.hasOwnProperty("dimParam"))
                        writer.uint32(/* id 2, wireType 2 =*/18).string(message.dimParam);
                    return writer;
                };
    
                /**
                 * Encodes the specified Dimension message, length delimited. Does not implicitly {@link onnx.TensorShapeProto.Dimension.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof onnx.TensorShapeProto.Dimension
                 * @static
                 * @param {onnx.TensorShapeProto.IDimension} message Dimension message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Dimension.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes a Dimension message from the specified reader or buffer.
                 * @function decode
                 * @memberof onnx.TensorShapeProto.Dimension
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {onnx.TensorShapeProto.Dimension} Dimension
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                Dimension.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TensorShapeProto.Dimension();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.dimValue = reader.int64();
                            break;
                        case 2:
                            message.dimParam = reader.string();
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                /**
                 * Decodes a Dimension message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof onnx.TensorShapeProto.Dimension
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {onnx.TensorShapeProto.Dimension} Dimension
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                Dimension.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies a Dimension message.
                 * @function verify
                 * @memberof onnx.TensorShapeProto.Dimension
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
                Dimension.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    var properties = {};
                    if (message.dimValue != null && message.hasOwnProperty("dimValue")) {
                        properties.value = 1;
                        if (!$util.isInteger(message.dimValue) && !(message.dimValue && $util.isInteger(message.dimValue.low) && $util.isInteger(message.dimValue.high)))
                            return "dimValue: integer|Long expected";
                    }
                    if (message.dimParam != null && message.hasOwnProperty("dimParam")) {
                        if (properties.value === 1)
                            return "value: multiple values";
                        properties.value = 1;
                        if (!$util.isString(message.dimParam))
                            return "dimParam: string expected";
                    }
                    return null;
                };
    
                /**
                 * Creates a Dimension message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof onnx.TensorShapeProto.Dimension
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {onnx.TensorShapeProto.Dimension} Dimension
                 */
                Dimension.fromObject = function fromObject(object) {
                    if (object instanceof $root.onnx.TensorShapeProto.Dimension)
                        return object;
                    var message = new $root.onnx.TensorShapeProto.Dimension();
                    if (object.dimValue != null)
                        if ($util.Long)
                            (message.dimValue = $util.Long.fromValue(object.dimValue)).unsigned = false;
                        else if (typeof object.dimValue === "string")
                            message.dimValue = parseInt(object.dimValue, 10);
                        else if (typeof object.dimValue === "number")
                            message.dimValue = object.dimValue;
                        else if (typeof object.dimValue === "object")
                            message.dimValue = new $util.LongBits(object.dimValue.low >>> 0, object.dimValue.high >>> 0).toNumber();
                    if (object.dimParam != null)
                        message.dimParam = String(object.dimParam);
                    return message;
                };
    
                /**
                 * Creates a plain object from a Dimension message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof onnx.TensorShapeProto.Dimension
                 * @static
                 * @param {onnx.TensorShapeProto.Dimension} message Dimension
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
                Dimension.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (message.dimValue != null && message.hasOwnProperty("dimValue")) {
                        if (typeof message.dimValue === "number")
                            object.dimValue = options.longs === String ? String(message.dimValue) : message.dimValue;
                        else
                            object.dimValue = options.longs === String ? $util.Long.prototype.toString.call(message.dimValue) : options.longs === Number ? new $util.LongBits(message.dimValue.low >>> 0, message.dimValue.high >>> 0).toNumber() : message.dimValue;
                        if (options.oneofs)
                            object.value = "dimValue";
                    }
                    if (message.dimParam != null && message.hasOwnProperty("dimParam")) {
                        object.dimParam = message.dimParam;
                        if (options.oneofs)
                            object.value = "dimParam";
                    }
                    return object;
                };
    
                /**
                 * Converts this Dimension to JSON.
                 * @function toJSON
                 * @memberof onnx.TensorShapeProto.Dimension
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                Dimension.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return Dimension;
            })();
    
            return TensorShapeProto;
        })();
    
        onnx.TypeProto = (function() {
    
            /**
             * Properties of a TypeProto.
             * @memberof onnx
             * @interface ITypeProto
             * @property {onnx.TypeProto.ITensor|null} [tensorType] TypeProto tensorType
             * @property {onnx.TypeProto.ISequence|null} [sequenceType] TypeProto sequenceType
             * @property {onnx.TypeProto.IMap|null} [mapType] TypeProto mapType
             */
    
            /**
             * Constructs a new TypeProto.
             * @memberof onnx
             * @classdesc Represents a TypeProto.
             * @implements ITypeProto
             * @constructor
             * @param {onnx.ITypeProto=} [properties] Properties to set
             */
            function TypeProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * TypeProto tensorType.
             * @member {onnx.TypeProto.ITensor|null|undefined} tensorType
             * @memberof onnx.TypeProto
             * @instance
             */
            TypeProto.prototype.tensorType = null;
    
            /**
             * TypeProto sequenceType.
             * @member {onnx.TypeProto.ISequence|null|undefined} sequenceType
             * @memberof onnx.TypeProto
             * @instance
             */
            TypeProto.prototype.sequenceType = null;
    
            /**
             * TypeProto mapType.
             * @member {onnx.TypeProto.IMap|null|undefined} mapType
             * @memberof onnx.TypeProto
             * @instance
             */
            TypeProto.prototype.mapType = null;
    
            // OneOf field names bound to virtual getters and setters
            var $oneOfFields;
    
            /**
             * TypeProto value.
             * @member {"tensorType"|"sequenceType"|"mapType"|undefined} value
             * @memberof onnx.TypeProto
             * @instance
             */
            Object.defineProperty(TypeProto.prototype, "value", {
                get: $util.oneOfGetter($oneOfFields = ["tensorType", "sequenceType", "mapType"]),
                set: $util.oneOfSetter($oneOfFields)
            });
    
            /**
             * Creates a new TypeProto instance using the specified properties.
             * @function create
             * @memberof onnx.TypeProto
             * @static
             * @param {onnx.ITypeProto=} [properties] Properties to set
             * @returns {onnx.TypeProto} TypeProto instance
             */
            TypeProto.create = function create(properties) {
                return new TypeProto(properties);
            };
    
            /**
             * Encodes the specified TypeProto message. Does not implicitly {@link onnx.TypeProto.verify|verify} messages.
             * @function encode
             * @memberof onnx.TypeProto
             * @static
             * @param {onnx.ITypeProto} message TypeProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TypeProto.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.tensorType != null && message.hasOwnProperty("tensorType"))
                    $root.onnx.TypeProto.Tensor.encode(message.tensorType, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                if (message.sequenceType != null && message.hasOwnProperty("sequenceType"))
                    $root.onnx.TypeProto.Sequence.encode(message.sequenceType, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
                if (message.mapType != null && message.hasOwnProperty("mapType"))
                    $root.onnx.TypeProto.Map.encode(message.mapType, writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
                return writer;
            };
    
            /**
             * Encodes the specified TypeProto message, length delimited. Does not implicitly {@link onnx.TypeProto.verify|verify} messages.
             * @function encodeDelimited
             * @memberof onnx.TypeProto
             * @static
             * @param {onnx.ITypeProto} message TypeProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TypeProto.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes a TypeProto message from the specified reader or buffer.
             * @function decode
             * @memberof onnx.TypeProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {onnx.TypeProto} TypeProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            TypeProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.tensorType = $root.onnx.TypeProto.Tensor.decode(reader, reader.uint32());
                        break;
                    case 4:
                        message.sequenceType = $root.onnx.TypeProto.Sequence.decode(reader, reader.uint32());
                        break;
                    case 5:
                        message.mapType = $root.onnx.TypeProto.Map.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            /**
             * Decodes a TypeProto message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof onnx.TypeProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {onnx.TypeProto} TypeProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            TypeProto.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies a TypeProto message.
             * @function verify
             * @memberof onnx.TypeProto
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            TypeProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                var properties = {};
                if (message.tensorType != null && message.hasOwnProperty("tensorType")) {
                    properties.value = 1;
                    {
                        var error = $root.onnx.TypeProto.Tensor.verify(message.tensorType);
                        if (error)
                            return "tensorType." + error;
                    }
                }
                if (message.sequenceType != null && message.hasOwnProperty("sequenceType")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    {
                        var error = $root.onnx.TypeProto.Sequence.verify(message.sequenceType);
                        if (error)
                            return "sequenceType." + error;
                    }
                }
                if (message.mapType != null && message.hasOwnProperty("mapType")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    {
                        var error = $root.onnx.TypeProto.Map.verify(message.mapType);
                        if (error)
                            return "mapType." + error;
                    }
                }
                return null;
            };
    
            /**
             * Creates a TypeProto message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof onnx.TypeProto
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {onnx.TypeProto} TypeProto
             */
            TypeProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.TypeProto)
                    return object;
                var message = new $root.onnx.TypeProto();
                if (object.tensorType != null) {
                    if (typeof object.tensorType !== "object")
                        throw TypeError(".onnx.TypeProto.tensorType: object expected");
                    message.tensorType = $root.onnx.TypeProto.Tensor.fromObject(object.tensorType);
                }
                if (object.sequenceType != null) {
                    if (typeof object.sequenceType !== "object")
                        throw TypeError(".onnx.TypeProto.sequenceType: object expected");
                    message.sequenceType = $root.onnx.TypeProto.Sequence.fromObject(object.sequenceType);
                }
                if (object.mapType != null) {
                    if (typeof object.mapType !== "object")
                        throw TypeError(".onnx.TypeProto.mapType: object expected");
                    message.mapType = $root.onnx.TypeProto.Map.fromObject(object.mapType);
                }
                return message;
            };
    
            /**
             * Creates a plain object from a TypeProto message. Also converts values to other types if specified.
             * @function toObject
             * @memberof onnx.TypeProto
             * @static
             * @param {onnx.TypeProto} message TypeProto
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            TypeProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (message.tensorType != null && message.hasOwnProperty("tensorType")) {
                    object.tensorType = $root.onnx.TypeProto.Tensor.toObject(message.tensorType, options);
                    if (options.oneofs)
                        object.value = "tensorType";
                }
                if (message.sequenceType != null && message.hasOwnProperty("sequenceType")) {
                    object.sequenceType = $root.onnx.TypeProto.Sequence.toObject(message.sequenceType, options);
                    if (options.oneofs)
                        object.value = "sequenceType";
                }
                if (message.mapType != null && message.hasOwnProperty("mapType")) {
                    object.mapType = $root.onnx.TypeProto.Map.toObject(message.mapType, options);
                    if (options.oneofs)
                        object.value = "mapType";
                }
                return object;
            };
    
            /**
             * Converts this TypeProto to JSON.
             * @function toJSON
             * @memberof onnx.TypeProto
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            TypeProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            TypeProto.Tensor = (function() {
    
                /**
                 * Properties of a Tensor.
                 * @memberof onnx.TypeProto
                 * @interface ITensor
                 * @property {onnx.TensorProto.DataType|null} [elemType] Tensor elemType
                 * @property {onnx.ITensorShapeProto|null} [shape] Tensor shape
                 */
    
                /**
                 * Constructs a new Tensor.
                 * @memberof onnx.TypeProto
                 * @classdesc Represents a Tensor.
                 * @implements ITensor
                 * @constructor
                 * @param {onnx.TypeProto.ITensor=} [properties] Properties to set
                 */
                function Tensor(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * Tensor elemType.
                 * @member {onnx.TensorProto.DataType} elemType
                 * @memberof onnx.TypeProto.Tensor
                 * @instance
                 */
                Tensor.prototype.elemType = 0;
    
                /**
                 * Tensor shape.
                 * @member {onnx.ITensorShapeProto|null|undefined} shape
                 * @memberof onnx.TypeProto.Tensor
                 * @instance
                 */
                Tensor.prototype.shape = null;
    
                /**
                 * Creates a new Tensor instance using the specified properties.
                 * @function create
                 * @memberof onnx.TypeProto.Tensor
                 * @static
                 * @param {onnx.TypeProto.ITensor=} [properties] Properties to set
                 * @returns {onnx.TypeProto.Tensor} Tensor instance
                 */
                Tensor.create = function create(properties) {
                    return new Tensor(properties);
                };
    
                /**
                 * Encodes the specified Tensor message. Does not implicitly {@link onnx.TypeProto.Tensor.verify|verify} messages.
                 * @function encode
                 * @memberof onnx.TypeProto.Tensor
                 * @static
                 * @param {onnx.TypeProto.ITensor} message Tensor message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Tensor.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.elemType != null && message.hasOwnProperty("elemType"))
                        writer.uint32(/* id 1, wireType 0 =*/8).int32(message.elemType);
                    if (message.shape != null && message.hasOwnProperty("shape"))
                        $root.onnx.TensorShapeProto.encode(message.shape, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                    return writer;
                };
    
                /**
                 * Encodes the specified Tensor message, length delimited. Does not implicitly {@link onnx.TypeProto.Tensor.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof onnx.TypeProto.Tensor
                 * @static
                 * @param {onnx.TypeProto.ITensor} message Tensor message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Tensor.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes a Tensor message from the specified reader or buffer.
                 * @function decode
                 * @memberof onnx.TypeProto.Tensor
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {onnx.TypeProto.Tensor} Tensor
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                Tensor.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto.Tensor();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.elemType = reader.int32();
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
    
                /**
                 * Decodes a Tensor message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof onnx.TypeProto.Tensor
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {onnx.TypeProto.Tensor} Tensor
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                Tensor.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies a Tensor message.
                 * @function verify
                 * @memberof onnx.TypeProto.Tensor
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
                Tensor.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.elemType != null && message.hasOwnProperty("elemType"))
                        switch (message.elemType) {
                        default:
                            return "elemType: enum value expected";
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
                            break;
                        }
                    if (message.shape != null && message.hasOwnProperty("shape")) {
                        var error = $root.onnx.TensorShapeProto.verify(message.shape);
                        if (error)
                            return "shape." + error;
                    }
                    return null;
                };
    
                /**
                 * Creates a Tensor message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof onnx.TypeProto.Tensor
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {onnx.TypeProto.Tensor} Tensor
                 */
                Tensor.fromObject = function fromObject(object) {
                    if (object instanceof $root.onnx.TypeProto.Tensor)
                        return object;
                    var message = new $root.onnx.TypeProto.Tensor();
                    switch (object.elemType) {
                    case "UNDEFINED":
                    case 0:
                        message.elemType = 0;
                        break;
                    case "FLOAT":
                    case 1:
                        message.elemType = 1;
                        break;
                    case "UINT8":
                    case 2:
                        message.elemType = 2;
                        break;
                    case "INT8":
                    case 3:
                        message.elemType = 3;
                        break;
                    case "UINT16":
                    case 4:
                        message.elemType = 4;
                        break;
                    case "INT16":
                    case 5:
                        message.elemType = 5;
                        break;
                    case "INT32":
                    case 6:
                        message.elemType = 6;
                        break;
                    case "INT64":
                    case 7:
                        message.elemType = 7;
                        break;
                    case "STRING":
                    case 8:
                        message.elemType = 8;
                        break;
                    case "BOOL":
                    case 9:
                        message.elemType = 9;
                        break;
                    case "FLOAT16":
                    case 10:
                        message.elemType = 10;
                        break;
                    case "DOUBLE":
                    case 11:
                        message.elemType = 11;
                        break;
                    case "UINT32":
                    case 12:
                        message.elemType = 12;
                        break;
                    case "UINT64":
                    case 13:
                        message.elemType = 13;
                        break;
                    case "COMPLEX64":
                    case 14:
                        message.elemType = 14;
                        break;
                    case "COMPLEX128":
                    case 15:
                        message.elemType = 15;
                        break;
                    }
                    if (object.shape != null) {
                        if (typeof object.shape !== "object")
                            throw TypeError(".onnx.TypeProto.Tensor.shape: object expected");
                        message.shape = $root.onnx.TensorShapeProto.fromObject(object.shape);
                    }
                    return message;
                };
    
                /**
                 * Creates a plain object from a Tensor message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof onnx.TypeProto.Tensor
                 * @static
                 * @param {onnx.TypeProto.Tensor} message Tensor
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
                Tensor.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        object.elemType = options.enums === String ? "UNDEFINED" : 0;
                        object.shape = null;
                    }
                    if (message.elemType != null && message.hasOwnProperty("elemType"))
                        object.elemType = options.enums === String ? $root.onnx.TensorProto.DataType[message.elemType] : message.elemType;
                    if (message.shape != null && message.hasOwnProperty("shape"))
                        object.shape = $root.onnx.TensorShapeProto.toObject(message.shape, options);
                    return object;
                };
    
                /**
                 * Converts this Tensor to JSON.
                 * @function toJSON
                 * @memberof onnx.TypeProto.Tensor
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                Tensor.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return Tensor;
            })();
    
            TypeProto.Sequence = (function() {
    
                /**
                 * Properties of a Sequence.
                 * @memberof onnx.TypeProto
                 * @interface ISequence
                 * @property {onnx.ITypeProto|null} [elemType] Sequence elemType
                 */
    
                /**
                 * Constructs a new Sequence.
                 * @memberof onnx.TypeProto
                 * @classdesc Represents a Sequence.
                 * @implements ISequence
                 * @constructor
                 * @param {onnx.TypeProto.ISequence=} [properties] Properties to set
                 */
                function Sequence(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * Sequence elemType.
                 * @member {onnx.ITypeProto|null|undefined} elemType
                 * @memberof onnx.TypeProto.Sequence
                 * @instance
                 */
                Sequence.prototype.elemType = null;
    
                /**
                 * Creates a new Sequence instance using the specified properties.
                 * @function create
                 * @memberof onnx.TypeProto.Sequence
                 * @static
                 * @param {onnx.TypeProto.ISequence=} [properties] Properties to set
                 * @returns {onnx.TypeProto.Sequence} Sequence instance
                 */
                Sequence.create = function create(properties) {
                    return new Sequence(properties);
                };
    
                /**
                 * Encodes the specified Sequence message. Does not implicitly {@link onnx.TypeProto.Sequence.verify|verify} messages.
                 * @function encode
                 * @memberof onnx.TypeProto.Sequence
                 * @static
                 * @param {onnx.TypeProto.ISequence} message Sequence message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Sequence.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.elemType != null && message.hasOwnProperty("elemType"))
                        $root.onnx.TypeProto.encode(message.elemType, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                    return writer;
                };
    
                /**
                 * Encodes the specified Sequence message, length delimited. Does not implicitly {@link onnx.TypeProto.Sequence.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof onnx.TypeProto.Sequence
                 * @static
                 * @param {onnx.TypeProto.ISequence} message Sequence message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Sequence.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes a Sequence message from the specified reader or buffer.
                 * @function decode
                 * @memberof onnx.TypeProto.Sequence
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {onnx.TypeProto.Sequence} Sequence
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                Sequence.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto.Sequence();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.elemType = $root.onnx.TypeProto.decode(reader, reader.uint32());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                /**
                 * Decodes a Sequence message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof onnx.TypeProto.Sequence
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {onnx.TypeProto.Sequence} Sequence
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                Sequence.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies a Sequence message.
                 * @function verify
                 * @memberof onnx.TypeProto.Sequence
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
                Sequence.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.elemType != null && message.hasOwnProperty("elemType")) {
                        var error = $root.onnx.TypeProto.verify(message.elemType);
                        if (error)
                            return "elemType." + error;
                    }
                    return null;
                };
    
                /**
                 * Creates a Sequence message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof onnx.TypeProto.Sequence
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {onnx.TypeProto.Sequence} Sequence
                 */
                Sequence.fromObject = function fromObject(object) {
                    if (object instanceof $root.onnx.TypeProto.Sequence)
                        return object;
                    var message = new $root.onnx.TypeProto.Sequence();
                    if (object.elemType != null) {
                        if (typeof object.elemType !== "object")
                            throw TypeError(".onnx.TypeProto.Sequence.elemType: object expected");
                        message.elemType = $root.onnx.TypeProto.fromObject(object.elemType);
                    }
                    return message;
                };
    
                /**
                 * Creates a plain object from a Sequence message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof onnx.TypeProto.Sequence
                 * @static
                 * @param {onnx.TypeProto.Sequence} message Sequence
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
                Sequence.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults)
                        object.elemType = null;
                    if (message.elemType != null && message.hasOwnProperty("elemType"))
                        object.elemType = $root.onnx.TypeProto.toObject(message.elemType, options);
                    return object;
                };
    
                /**
                 * Converts this Sequence to JSON.
                 * @function toJSON
                 * @memberof onnx.TypeProto.Sequence
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                Sequence.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return Sequence;
            })();
    
            TypeProto.Map = (function() {
    
                /**
                 * Properties of a Map.
                 * @memberof onnx.TypeProto
                 * @interface IMap
                 * @property {onnx.TensorProto.DataType|null} [keyType] Map keyType
                 * @property {onnx.ITypeProto|null} [valueType] Map valueType
                 */
    
                /**
                 * Constructs a new Map.
                 * @memberof onnx.TypeProto
                 * @classdesc Represents a Map.
                 * @implements IMap
                 * @constructor
                 * @param {onnx.TypeProto.IMap=} [properties] Properties to set
                 */
                function Map(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                /**
                 * Map keyType.
                 * @member {onnx.TensorProto.DataType} keyType
                 * @memberof onnx.TypeProto.Map
                 * @instance
                 */
                Map.prototype.keyType = 0;
    
                /**
                 * Map valueType.
                 * @member {onnx.ITypeProto|null|undefined} valueType
                 * @memberof onnx.TypeProto.Map
                 * @instance
                 */
                Map.prototype.valueType = null;
    
                /**
                 * Creates a new Map instance using the specified properties.
                 * @function create
                 * @memberof onnx.TypeProto.Map
                 * @static
                 * @param {onnx.TypeProto.IMap=} [properties] Properties to set
                 * @returns {onnx.TypeProto.Map} Map instance
                 */
                Map.create = function create(properties) {
                    return new Map(properties);
                };
    
                /**
                 * Encodes the specified Map message. Does not implicitly {@link onnx.TypeProto.Map.verify|verify} messages.
                 * @function encode
                 * @memberof onnx.TypeProto.Map
                 * @static
                 * @param {onnx.TypeProto.IMap} message Map message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Map.encode = function encode(message, writer) {
                    if (!writer)
                        writer = $Writer.create();
                    if (message.keyType != null && message.hasOwnProperty("keyType"))
                        writer.uint32(/* id 1, wireType 0 =*/8).int32(message.keyType);
                    if (message.valueType != null && message.hasOwnProperty("valueType"))
                        $root.onnx.TypeProto.encode(message.valueType, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                    return writer;
                };
    
                /**
                 * Encodes the specified Map message, length delimited. Does not implicitly {@link onnx.TypeProto.Map.verify|verify} messages.
                 * @function encodeDelimited
                 * @memberof onnx.TypeProto.Map
                 * @static
                 * @param {onnx.TypeProto.IMap} message Map message or plain object to encode
                 * @param {$protobuf.Writer} [writer] Writer to encode to
                 * @returns {$protobuf.Writer} Writer
                 */
                Map.encodeDelimited = function encodeDelimited(message, writer) {
                    return this.encode(message, writer).ldelim();
                };
    
                /**
                 * Decodes a Map message from the specified reader or buffer.
                 * @function decode
                 * @memberof onnx.TypeProto.Map
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @param {number} [length] Message length if known beforehand
                 * @returns {onnx.TypeProto.Map} Map
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                Map.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto.Map();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.keyType = reader.int32();
                            break;
                        case 2:
                            message.valueType = $root.onnx.TypeProto.decode(reader, reader.uint32());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                /**
                 * Decodes a Map message from the specified reader or buffer, length delimited.
                 * @function decodeDelimited
                 * @memberof onnx.TypeProto.Map
                 * @static
                 * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
                 * @returns {onnx.TypeProto.Map} Map
                 * @throws {Error} If the payload is not a reader or valid buffer
                 * @throws {$protobuf.util.ProtocolError} If required fields are missing
                 */
                Map.decodeDelimited = function decodeDelimited(reader) {
                    if (!(reader instanceof $Reader))
                        reader = new $Reader(reader);
                    return this.decode(reader, reader.uint32());
                };
    
                /**
                 * Verifies a Map message.
                 * @function verify
                 * @memberof onnx.TypeProto.Map
                 * @static
                 * @param {Object.<string,*>} message Plain object to verify
                 * @returns {string|null} `null` if valid, otherwise the reason why it is not
                 */
                Map.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.keyType != null && message.hasOwnProperty("keyType"))
                        switch (message.keyType) {
                        default:
                            return "keyType: enum value expected";
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
                            break;
                        }
                    if (message.valueType != null && message.hasOwnProperty("valueType")) {
                        var error = $root.onnx.TypeProto.verify(message.valueType);
                        if (error)
                            return "valueType." + error;
                    }
                    return null;
                };
    
                /**
                 * Creates a Map message from a plain object. Also converts values to their respective internal types.
                 * @function fromObject
                 * @memberof onnx.TypeProto.Map
                 * @static
                 * @param {Object.<string,*>} object Plain object
                 * @returns {onnx.TypeProto.Map} Map
                 */
                Map.fromObject = function fromObject(object) {
                    if (object instanceof $root.onnx.TypeProto.Map)
                        return object;
                    var message = new $root.onnx.TypeProto.Map();
                    switch (object.keyType) {
                    case "UNDEFINED":
                    case 0:
                        message.keyType = 0;
                        break;
                    case "FLOAT":
                    case 1:
                        message.keyType = 1;
                        break;
                    case "UINT8":
                    case 2:
                        message.keyType = 2;
                        break;
                    case "INT8":
                    case 3:
                        message.keyType = 3;
                        break;
                    case "UINT16":
                    case 4:
                        message.keyType = 4;
                        break;
                    case "INT16":
                    case 5:
                        message.keyType = 5;
                        break;
                    case "INT32":
                    case 6:
                        message.keyType = 6;
                        break;
                    case "INT64":
                    case 7:
                        message.keyType = 7;
                        break;
                    case "STRING":
                    case 8:
                        message.keyType = 8;
                        break;
                    case "BOOL":
                    case 9:
                        message.keyType = 9;
                        break;
                    case "FLOAT16":
                    case 10:
                        message.keyType = 10;
                        break;
                    case "DOUBLE":
                    case 11:
                        message.keyType = 11;
                        break;
                    case "UINT32":
                    case 12:
                        message.keyType = 12;
                        break;
                    case "UINT64":
                    case 13:
                        message.keyType = 13;
                        break;
                    case "COMPLEX64":
                    case 14:
                        message.keyType = 14;
                        break;
                    case "COMPLEX128":
                    case 15:
                        message.keyType = 15;
                        break;
                    }
                    if (object.valueType != null) {
                        if (typeof object.valueType !== "object")
                            throw TypeError(".onnx.TypeProto.Map.valueType: object expected");
                        message.valueType = $root.onnx.TypeProto.fromObject(object.valueType);
                    }
                    return message;
                };
    
                /**
                 * Creates a plain object from a Map message. Also converts values to other types if specified.
                 * @function toObject
                 * @memberof onnx.TypeProto.Map
                 * @static
                 * @param {onnx.TypeProto.Map} message Map
                 * @param {$protobuf.IConversionOptions} [options] Conversion options
                 * @returns {Object.<string,*>} Plain object
                 */
                Map.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        object.keyType = options.enums === String ? "UNDEFINED" : 0;
                        object.valueType = null;
                    }
                    if (message.keyType != null && message.hasOwnProperty("keyType"))
                        object.keyType = options.enums === String ? $root.onnx.TensorProto.DataType[message.keyType] : message.keyType;
                    if (message.valueType != null && message.hasOwnProperty("valueType"))
                        object.valueType = $root.onnx.TypeProto.toObject(message.valueType, options);
                    return object;
                };
    
                /**
                 * Converts this Map to JSON.
                 * @function toJSON
                 * @memberof onnx.TypeProto.Map
                 * @instance
                 * @returns {Object.<string,*>} JSON object
                 */
                Map.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return Map;
            })();
    
            return TypeProto;
        })();
    
        onnx.OperatorSetIdProto = (function() {
    
            /**
             * Properties of an OperatorSetIdProto.
             * @memberof onnx
             * @interface IOperatorSetIdProto
             * @property {string|null} [domain] OperatorSetIdProto domain
             * @property {number|Long|null} [version] OperatorSetIdProto version
             */
    
            /**
             * Constructs a new OperatorSetIdProto.
             * @memberof onnx
             * @classdesc Represents an OperatorSetIdProto.
             * @implements IOperatorSetIdProto
             * @constructor
             * @param {onnx.IOperatorSetIdProto=} [properties] Properties to set
             */
            function OperatorSetIdProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            /**
             * OperatorSetIdProto domain.
             * @member {string} domain
             * @memberof onnx.OperatorSetIdProto
             * @instance
             */
            OperatorSetIdProto.prototype.domain = "";
    
            /**
             * OperatorSetIdProto version.
             * @member {number|Long} version
             * @memberof onnx.OperatorSetIdProto
             * @instance
             */
            OperatorSetIdProto.prototype.version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
            /**
             * Creates a new OperatorSetIdProto instance using the specified properties.
             * @function create
             * @memberof onnx.OperatorSetIdProto
             * @static
             * @param {onnx.IOperatorSetIdProto=} [properties] Properties to set
             * @returns {onnx.OperatorSetIdProto} OperatorSetIdProto instance
             */
            OperatorSetIdProto.create = function create(properties) {
                return new OperatorSetIdProto(properties);
            };
    
            /**
             * Encodes the specified OperatorSetIdProto message. Does not implicitly {@link onnx.OperatorSetIdProto.verify|verify} messages.
             * @function encode
             * @memberof onnx.OperatorSetIdProto
             * @static
             * @param {onnx.IOperatorSetIdProto} message OperatorSetIdProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            OperatorSetIdProto.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.domain != null && message.hasOwnProperty("domain"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.domain);
                if (message.version != null && message.hasOwnProperty("version"))
                    writer.uint32(/* id 2, wireType 0 =*/16).int64(message.version);
                return writer;
            };
    
            /**
             * Encodes the specified OperatorSetIdProto message, length delimited. Does not implicitly {@link onnx.OperatorSetIdProto.verify|verify} messages.
             * @function encodeDelimited
             * @memberof onnx.OperatorSetIdProto
             * @static
             * @param {onnx.IOperatorSetIdProto} message OperatorSetIdProto message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            OperatorSetIdProto.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };
    
            /**
             * Decodes an OperatorSetIdProto message from the specified reader or buffer.
             * @function decode
             * @memberof onnx.OperatorSetIdProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {onnx.OperatorSetIdProto} OperatorSetIdProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
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
    
            /**
             * Decodes an OperatorSetIdProto message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof onnx.OperatorSetIdProto
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {onnx.OperatorSetIdProto} OperatorSetIdProto
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            OperatorSetIdProto.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };
    
            /**
             * Verifies an OperatorSetIdProto message.
             * @function verify
             * @memberof onnx.OperatorSetIdProto
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            OperatorSetIdProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.domain != null && message.hasOwnProperty("domain"))
                    if (!$util.isString(message.domain))
                        return "domain: string expected";
                if (message.version != null && message.hasOwnProperty("version"))
                    if (!$util.isInteger(message.version) && !(message.version && $util.isInteger(message.version.low) && $util.isInteger(message.version.high)))
                        return "version: integer|Long expected";
                return null;
            };
    
            /**
             * Creates an OperatorSetIdProto message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof onnx.OperatorSetIdProto
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {onnx.OperatorSetIdProto} OperatorSetIdProto
             */
            OperatorSetIdProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.OperatorSetIdProto)
                    return object;
                var message = new $root.onnx.OperatorSetIdProto();
                if (object.domain != null)
                    message.domain = String(object.domain);
                if (object.version != null)
                    if ($util.Long)
                        (message.version = $util.Long.fromValue(object.version)).unsigned = false;
                    else if (typeof object.version === "string")
                        message.version = parseInt(object.version, 10);
                    else if (typeof object.version === "number")
                        message.version = object.version;
                    else if (typeof object.version === "object")
                        message.version = new $util.LongBits(object.version.low >>> 0, object.version.high >>> 0).toNumber();
                return message;
            };
    
            /**
             * Creates a plain object from an OperatorSetIdProto message. Also converts values to other types if specified.
             * @function toObject
             * @memberof onnx.OperatorSetIdProto
             * @static
             * @param {onnx.OperatorSetIdProto} message OperatorSetIdProto
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            OperatorSetIdProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.domain = "";
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.version = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.version = options.longs === String ? "0" : 0;
                }
                if (message.domain != null && message.hasOwnProperty("domain"))
                    object.domain = message.domain;
                if (message.version != null && message.hasOwnProperty("version"))
                    if (typeof message.version === "number")
                        object.version = options.longs === String ? String(message.version) : message.version;
                    else
                        object.version = options.longs === String ? $util.Long.prototype.toString.call(message.version) : options.longs === Number ? new $util.LongBits(message.version.low >>> 0, message.version.high >>> 0).toNumber() : message.version;
                return object;
            };
    
            /**
             * Converts this OperatorSetIdProto to JSON.
             * @function toJSON
             * @memberof onnx.OperatorSetIdProto
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            OperatorSetIdProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return OperatorSetIdProto;
        })();
    
        return onnx;
    })();

    return $root;
})(protobuf);
