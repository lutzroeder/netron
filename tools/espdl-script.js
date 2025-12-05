import * as fs from 'fs/promises';
import * as path from 'path';
import * as url from 'url';

const main = async () => {
    const dirname = path.dirname(url.fileURLToPath(import.meta.url));
    const file = path.join(dirname, '..', 'source', 'espdl-metadata.json');

    // Read existing metadata
    const input = await fs.readFile(file, 'utf-8');
    const json = JSON.parse(input);

    // Validate and process operators
    const operators = new Map();
    const errors = [];

    for (const operator of json) {
        // Check for duplicate operator names
        if (operators.has(operator.name)) {
            errors.push(`Duplicate operator '${operator.name}'.`);
        }
        operators.set(operator.name, operator);

        // Validate required fields
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

        // Validate inputs
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

        // Validate outputs
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

    // Report any errors
    if (errors.length > 0) {
        throw new Error(`ESPDL metadata validation errors:\n${errors.join('\n')}`);
    }

    // Sort operators by name
    const beforeSort = json.map(op => op.name);
    json.sort((a, b) => a.name.localeCompare(b.name));
    const afterSort = json.map(op => op.name);

    // Check if sorting changed anything
    let changed = false;
    for (let i = 0; i < beforeSort.length; i++) {
        if (beforeSort[i] !== afterSort[i]) {
            changed = true;
            break;
        }
    }

    if (changed) {
        console.log(`Sorted ${json.length} operators.`);
    }

    // Format JSON with consistent indentation
    let output = JSON.stringify(json, null, 2);

    // Apply formatting similar to other metadata files
    output = output.replace(/\s {8}/g, ' ');
    output = output.replace(/,\s {8}/g, ', ');
    output = output.replace(/\s {6}}/g, ' }');

    // Write back to file
    await fs.writeFile(file, output, 'utf-8');

    console.log(`Processed ${json.length} ESPDL operators.`);
};

await main().catch((error) => {
    console.error(error.message);
    process.exit(1);
});