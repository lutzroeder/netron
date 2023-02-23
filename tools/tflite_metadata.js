
const path = require('path');
const flatc = require('./flatc');
const fs = require('fs');

const schema = path.join(__dirname, '..', 'third_party', 'source', 'tensorflow', 'tensorflow', 'lite', 'schema', 'schema.fbs');
const file = path.join(__dirname, '..', 'source', 'tflite-metadata.json');

const input = fs.readFileSync(file, 'utf-8');
const json = JSON.parse(input);

const operators = new Map();
const attributes = new Map();
for (const operator of json) {
    if (operators.has(operator.name)) {
        throw new Error("Duplicate operator '" + operator.name + "'.");
    }
    operators.set(operator.name, operator);
    if (operator && operator.attributes) {
        for (const attribute of operator.attributes) {
            const name = operator.name + ':' + attribute.name;
            attributes.set(name, attribute);
        }
    }
}

const root = new flatc.Root('tflite', [], [ schema ]);
const namespace = root.find('tflite', flatc.Namespace);

const builtOperator = namespace.find('tflite.BuiltinOperator', flatc.Type);
const upperCase = new Set([ '2D', 'LSH', 'SVDF', 'RNN', 'L2', 'LSTM' ]);
for (const op of builtOperator.values.keys()) {
    let op_key = op === 'BATCH_MATMUL' ? 'BATCH_MAT_MUL' : op;
    op_key = op_key.split('_').map((s) => (s.length < 1 || upperCase.has(s)) ? s : s[0] + s.substring(1).toLowerCase()).join('');
    const table = namespace.find('tflite.' + op_key + 'Options', flatc.Type);
    if (table && table.fields.size > 0) {
        if (!operators.has(op_key)) {
            const operator = { name: op_key };
            operators.set(op_key, operator);
            json.push(operator);
        }
        const operator = operators.get(op_key);
        operator.attributes = operator.attributes || [];
        for (const field of table.fields.values()) {
            const attr_key = op_key + ':' + field.name;
            if (!attributes.has(attr_key)) {
                const attribute = { name: field.name };
                attributes.set(attr_key, attribute);
                operator.attributes.push(attribute);
            }
            const attribute = attributes.get(attr_key);
            const type = field.type;
            let defaultValue = field.defaultValue;
            if (type instanceof flatc.Enum) {
                if (!type.keys.has(defaultValue)) {
                    throw new Error("Invalid '" + type.name + "' default value '" + defaultValue + "'.");
                }
                defaultValue = type.keys.get(defaultValue);
            }
            attribute.type = type.name === 'bool' ? 'boolean' : type.name + (field.repeated ? '[]' : '');
            attribute.default = defaultValue;
        }
    }
}

json.sort((a, b) => a.name.localeCompare(b.name));

let output = JSON.stringify(json, null, 2);
output = output.replace(/\s {8}/g, ' ');
output = output.replace(/,\s {8}/g, ', ');
output = output.replace(/\s {6}}/g, ' }');
fs.writeFileSync(file, output, 'utf-8');
