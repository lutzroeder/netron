/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $util = $protobuf.util;
    
    var $root = $protobuf.roots.uff || ($protobuf.roots.uff = {});
    
    $root.uff = (function() {
    
        var uff = {};
    
        uff.MetaGraph = (function() {
    
            function MetaGraph(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            MetaGraph.prototype.version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            MetaGraph.prototype.descriptor_core_version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
            MetaGraph.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.uff.MetaGraph();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.version = reader.int64();
                        break;
                    case 2:
                        message.descriptor_core_version = reader.int64();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            MetaGraph.decodeText = function decodeText(reader) {
                var message = new $root.uff.MetaGraph();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "version":
                        message.version = reader.int64();
                        break;
                    case "descriptor_core_version":
                        message.descriptor_core_version = reader.int64();
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return MetaGraph;
        })();
    
        return uff;
    })();

    return $root;
})(protobuf);
