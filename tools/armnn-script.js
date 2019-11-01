var flatbuffers = require('../third_party/flatbuffers/js/flatbuffers.js').flatbuffers;
var schema = require('../src/armnn-schema.js');

console.log('');
console.log('armnnSerializer.castLayer = function(schema, layer) {');
console.log('    let layerType = layer.layerType();');

for (k of Object.keys(schema.armnnSerializer.Layer)) {
    if (k == "NONE")
        continue;

    console.log("    if (layerType == schema.Layer." + k + ")");
    console.log("        return layer.layer(new schema." + k + ");");
}

console.log('    return null;');
console.log('}')
