
import * as fs from 'fs/promises';
import * as path from 'path';
import * as url from 'url';

const main = async () => {
    const dirname = path.dirname(url.fileURLToPath(import.meta.url));
    const file = path.join(dirname, '..', 'source', 'espdl-metadata.json');
    const input = await fs.readFile(file, 'utf-8');
    const json = JSON.parse(input);
    const operators = new Map();
    const errors = [];
    for (const operator of json) {
        if (operators.has(operator.name)) {
            errors.push(`Duplicate operator '${operator.name}'.`);
        }
        operators.set(operator.name, operator);
        if (!operator.name) {
            errors.push(`Operator missing 'name' field.`);
        }
        if (!operator.module) {
            errors.push(`Operator '${operator.name}' missing 'module' field.`);
        }
        if (operator.version === undefined) {
            errors.push(`Operator '${operator.name}' missing 'version' field.`);
        }
        if (!operator.description) {
            errors.push(`Operator '${operator.name}' missing 'description' field.`);
        }
        if (operator.inputs && Array.isArray(operator.inputs)) {
            const inputNames = new Set();
            for (const input of operator.inputs) {
                if (!input.name) {
                    errors.push(`Operator '${operator.name}' has input missing 'name' field.`);
                }
                if (inputNames.has(input.name)) {
                    errors.push(`Operator '${operator.name}' has duplicate input name '${input.name}'.`);
                }
                inputNames.add(input.name);
            }
        }
        if (operator.outputs && Array.isArray(operator.outputs)) {
            const outputNames = new Set();
            for (const output of operator.outputs) {
                if (!output.name) {
                    errors.push(`Operator '${operator.name}' has output missing 'name' field.`);
                }
                if (outputNames.has(output.name)) {
                    errors.push(`Operator '${operator.name}' has duplicate output name '${output.name}'.`);
                }
                outputNames.add(output.name);
            }
        }
    }
    if (errors.length > 0) {
        throw new Error(`ESPDL metadata validation errors:\n${errors.join('\n')}`);
    }
    const beforeSort = json.map((op) => op.name);
    json.sort((a, b) => a.name.localeCompare(b.name));
    const afterSort = json.map((op) => op.name);
    for (let i = 0; i < beforeSort.length; i++) {
        if (beforeSort[i] !== afterSort[i]) {
            break;
        }
    }
    let output = JSON.stringify(json, null, 2);
    output = output.replace(/\s {8}/g, ' ');
    output = output.replace(/,\s {8}/g, ', ');
    output = output.replace(/\s {6}}/g, ' }');
    await fs.writeFile(file, output, 'utf-8');
};

await main().catch((error) => {
    // eslint-disable-next-line no-console
    console.error(error.message);
    process.exit(1);
});