
import * as flatc from './flatc.js';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as url from 'url';

/* eslint-disable no-extend-native */

BigInt.prototype.toNumber = function() {
    if (this > Number.MAX_SAFE_INTEGER || this < Number.MIN_SAFE_INTEGER) {
        throw new Error('64-bit value exceeds safe integer.');
    }
    return Number(this);
};

/* eslint-enable no-extend-native */

const main = async () => {
    const dirname = path.dirname(url.fileURLToPath(import.meta.url));
    const schema = path.join(dirname, '..', 'third_party', 'source', 'circle', 'nnpackage', 'schema', 'circle_schema.fbs');
    const file = path.join(dirname, '..', 'source', 'circle-metadata.json');
    const input = await fs.readFile(file, 'utf-8');
    const json = JSON.parse(input);
    const operators = new Map();
    const attributes = new Map();
    for (const operator of json) {
        if (operators.has(operator.name)) {
            throw new Error(`Duplicate operator '${operator.name}'.`);
        }
        operators.set(operator.name, operator);
        if (operator && operator.attributes) {
            for (const attribute of operator.attributes) {
                const name = `${operator.name}:${attribute.name}`;
                attributes.set(name, attribute);
            }
        }
    }
    const root = new flatc.Root('circle');
    await root.load([], [schema]);
    const namespace = root.find('circle', flatc.Namespace);
    const builtOperator = namespace.find('circle.BuiltinOperator', flatc.Type);
    const upperCase = new Set(['2D', 'LSH', 'SVDF', 'RNN', 'L2', 'LSTM']);
    for (const op of builtOperator.values.keys()) {
        let op_key = op === 'BATCH_MATMUL' ? 'BATCH_MAT_MUL' : op;
        op_key = op_key.split('_').map((s) => (s.length < 1 || upperCase.has(s)) ? s : s[0] + s.substring(1).toLowerCase()).join('');
        const table = namespace.find(`circle.${op_key}Options`, flatc.Type);
        if (table && table.fields.size > 0) {
            if (!operators.has(op_key)) {
                const operator = { name: op_key };
                operators.set(op_key, operator);
                json.push(operator);
            }
            const operator = operators.get(op_key);
            operator.attributes = operator.attributes || [];
            for (const field of table.fields.values()) {
                const attr_key = `${op_key}:${field.name}`;
                if (!attributes.has(attr_key)) {
                    const attribute = { name: field.name };
                    attributes.set(attr_key, attribute);
                    operator.attributes.push(attribute);
                }
                const attribute = attributes.get(attr_key);
                const type = field.type;
                let defaultValue = field.defaultValue;
                if (typeof defaultValue === 'bigint') {
                    defaultValue = defaultValue.toNumber();
                }
                if (type instanceof flatc.Enum) {
                    if (!type.keys.has(defaultValue)) {
                        throw new Error(`Invalid '${type.name}' default value '${defaultValue}'.`);
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
    await fs.writeFile(file, output, 'utf-8');
};

await main();
