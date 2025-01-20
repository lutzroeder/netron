
import * as flatc from './flatc.js';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as url from 'url';

const main = async () => {
    const dirname = path.dirname(url.fileURLToPath(import.meta.url));
    const schema = path.join(dirname, '..', 'third_party', 'source', 'mindspore', 'mindspore', 'lite', 'schema', 'ops.fbs');
    const file = path.join(dirname, '..', 'source', 'mslite-metadata.json');
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
    const root = new flatc.Root('mslite');
    await root.load([], [schema]);
    const namespace = root.find('mindspore.schema', flatc.Namespace);
    const primitiveType = namespace.find('mindspore.schema.PrimitiveType', flatc.Type);
    for (const value of primitiveType.values) {
        const table = value.type;
        const op_key = table.name;
        if (!operators.has(op_key)) {
            const operator = { name: op_key };
            operators.set(op_key, operator);
            json.push(operator);
        }
        const operator = operators.get(op_key);
        if (table && table.fields.size > 0) {
            operator.attributes = operator.attributes || [];
            const inputs = operator.inputs;
            const outputs = operator.outputs;
            delete operator.inputs;
            delete operator.outputs;
            if (inputs) {
                operator.inputs = inputs;
            }
            if (outputs) {
                operator.outputs = outputs;
            }
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
                if (type instanceof flatc.Enum) {
                    if (!type.keys.has(defaultValue)) {
                        throw new Error(`Invalid '${type.name}' default value '${defaultValue}'.`);
                    }
                    defaultValue = type.keys.get(defaultValue);
                }
                attribute.type = type.name === 'bool' ? 'boolean' : type.name + (field.repeated ? '[]' : '');
                if (attribute.default === undefined) {
                    attribute.default = defaultValue;
                }
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