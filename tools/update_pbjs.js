/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */
/* eslint "no-console": off */

const fs = require('fs');
const process = require('process');

const pattern = process.argv[2];
const file = process.argv[3];
const variable = process.argv[4];
const type = process.argv[5];
const count = parseInt(process.argv[6]);

let arrayType = '';
let dataViewMethod = '';
let shift = 0;

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

const source = fs.readFileSync(file, 'utf-8');

let search = '';
let replace = '';

switch (pattern) {
    case 'array':
        search = `if ((tag & 7) === 2) {
    var end2 = reader.uint32() + reader.pos;
    while (reader.pos < end2)
        message.$(variable).push(reader.$(type)());
} else`;
        replace = `if ((tag & 7) === 2) {
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
        break;

    case 'enumeration':
        search = `if (!(message.$(variable) && message.$(variable).length))
    message.$(variable) = [];
if ((tag & 7) === 2) {
    var end2 = reader.uint32() + reader.pos;
    while (reader.pos < end2)
        message.$(variable).push(reader.$(type)());
} else
    message.$(variable).push(reader.$(type)());
break;`;

        replace = `if (!(message.$(variable) && message.$(variable).length)) {
    if (message.$(variable) != -1) {
        message.$(variable) = [];
        message.$(variable)Count = 0;
    }
}
if (message.$(variable)Count < 1000000) {
    if ((tag & 7) === 2) {
        var end2 = reader.uint32() + reader.pos;
        while (reader.pos < end2) {
            message.$(variable).push(reader.$(type)());
            message.$(variable)Count++;
        }
    }
    else {
        message.$(variable).push(reader.$(type)());
        message.$(variable)Count++;
    }
}
else {
    message.$(variable) = -1;
    if ((tag & 7) === 2) {
        var endx = reader.uint32() + reader.pos;
        while (reader.pos < endx)
            reader.$(type)();
    }
    else {
        reader.$(type)();
    }
}
break;`;
        break;

    default:
        console.log('ERROR: Unknown pattern.')
        process.exit(1);
}

search = search.split('$(variable)').join(variable);
search = search.split('$(type)').join(type);

replace = replace.split('$(variable)').join(variable);
replace = replace.split('$(type)').join(type);
replace = replace.split('$(arrayType)').join(arrayType);
replace = replace.split('$(dataViewMethod)').join(dataViewMethod);
replace = replace.split('$(shift)').join(shift);

for (let i = 0; i < 8; i++) {

    search = search.split('\n').map((line) => '    ' + line).join('\n');
    replace = replace.split('\n').map((line) => '    ' + line).join('\n');

    const parts = source.split(search);
    if (parts.length == (count + 1)) {
        const target = parts.join(replace);
        fs.writeFileSync(file, target, 'utf-8');
        process.exit(0);
    }
}

console.log('ERROR: Replace failed.');
process.exit(1);
