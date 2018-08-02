/*jshint esversion: 6 */

const fs = require('fs');
const process = require('process');

var file = process.argv[2];
var variable = process.argv[3];
var type = process.argv[4];
var count = parseInt(process.argv[5]);

var arrayType = '';
var dataViewMethod = '';
var shift = 0;

switch (type) {
    case 'float':
        arrayType = 'Float32Array';
        dataViewMethod = 'getFloat32';
        shift = '2';
        break;
    case 'double':
        arrayType = 'Float64Array';
        dataViewMethod = 'getFloat64';
        shift = '3';
        break;
    default:
        console.log('ERROR: Type is not supported.');
        process.exit(1);
        break;
}

var source = fs.readFileSync(file, 'utf-8');

var search = `if ((tag & 7) === 2) {
    var end2 = reader.uint32() + reader.pos;
    while (reader.pos < end2)
        message.$(variable).push(reader.$(type)());
} else`;

var replace = `if ((tag & 7) === 2) {
    var end2 = reader.uint32() + reader.pos;
    if (message.$(variable).length == 0 && (end2 - reader.pos) > 1048576) {
        var $(variable)Length = end2 - reader.pos;
        var $(variable)View = new DataView(reader.buf.buffer, reader.buf.byteOffset + reader.pos, $(variable)Length);
        $(variable)Length = $(variable)Length >>> $(shift);
        var $(variable) = new $(arrayType)($(variable)Length);
        for (var i = 0; i < $(variable)Length; i++) {
            $(variable)[i] = $(variable)View.$(dataViewMethod)(i << $(shift), true);
        }
        message.$(variable) = $(variable);
        reader.pos = end2;
    }
    else {
        while (reader.pos < end2)
            message.$(variable).push(reader.$(type)());
    }
} else`;

search = search.split('$(variable)').join(variable);
search = search.split('$(type)').join(type);

replace = replace.split('$(variable)').join(variable);
replace = replace.split('$(type)').join(type);
replace = replace.split('$(arrayType)').join(arrayType);
replace = replace.split('$(dataViewMethod)').join(dataViewMethod);
replace = replace.split('$(shift)').join(shift);

for (var i = 0; i < 8; i++) {

    search = search.split('\n').map((line) => '    ' + line).join('\n');
    replace = replace.split('\n').map((line) => '    ' + line).join('\n');

    var parts = source.split(search);
    if (parts.length == (count + 1)) {
        var target = parts.join(replace);
        fs.writeFileSync(file, target, 'utf-8');
        process.exit(0);
    }
}

console.log('ERROR: Replace failed.');
process.exit(1);
