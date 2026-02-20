
import * as fs from 'fs/promises';
import * as path from 'path';
import * as url from 'url';
import * as xml from '../source/xml.js';

const types = new Map([
    ['i32_t', 'int32'], ['int32_t', 'int32'], ['i8_t', 'int8'], ['int16_t', 'int16'],
    ['bool_t', 'boolean'], ['String', 'string'],
    ['acc_type_t', 'DType'], ['acc_size_t', 'DType'], ['var_t', 'DType'],
    ['resize_mode_t', 'ResizeMode'], ['nan_propagation_mode_t', 'NanPropagationMode'], ['rounding_mode_t', 'RoundingMode'],
    ['shape_t', 'shape'], ['tensor_list_t', 'tensor[]'], ['tensor_size_t', 'int32[]'], ['tosa_graph_t', 'graph'],
    ['in_t', ''], ['out_t', ''], ['in_out_t', ''], ['weight_t', ''], ['mul_t', ''], ['index_t', ''], ['table_t', '']
]);

const categories = (name) => {
    if (name.includes('CONST')) {
        return 'Constant';
    }
    if (name === 'CUSTOM') {
        return 'Custom';
    }
    if (name.includes('POOL')) {
        return 'Pool';
    }
    if (name === 'RESCALE') {
        return 'Quantization';
    }
    if (name.includes('SHAPE') || name.includes('DIM')) {
        return 'Shape';
    }
    if (name.includes('CONV') || name === 'FULLY_CONNECTED') {
        return 'Layer';
    }
    if (name === 'TRANSPOSE' || name === 'GATHER' || name === 'SCATTER') {
        return 'Transform';
    }
    if (name === 'CLAMP' || name === 'SIGMOID' || name === 'TANH' || name === 'ERF') {
        return 'Activation';
    }
    if (name === 'CONCAT' || name === 'PAD' || name === 'REVERSE' || name === 'SLICE' || name === 'TILE' || name === 'IDENTITY') {
        return 'Tensor';
    }
    return undefined;
};

const main = async () => {
    const dirname = path.dirname(url.fileURLToPath(import.meta.url));
    const versions = [
        { version: '0.80', dir: 'v0.80' },
        { version: '1.0', dir: 'v1.0' }
    ];
    const files = await Promise.all(versions.map(({ dir }) => {
        const xmlPath = path.join(dirname, '..', 'third_party', 'source', 'tosa', dir, 'tosa.xml');
        return fs.readFile(xmlPath);
    }));
    const entries = [];
    for (let i = 0; i < versions.length; i++) {
        const version = versions[i].version;
        const reader = xml.TextReader.open(files[i]);
        const document = reader.read();
        const root = document.documentElement;
        for (const operatorsEl of root.getElementsByTagName('operators')) {
            for (const group of operatorsEl.getElementsByTagName('operatorgroup')) {
                for (const operator of group.getElementsByTagName('operator')) {
                    const nameElements = operator.getElementsByTagName('name');
                    if (nameElements.length === 0) {
                        continue;
                    }
                    const name = nameElements[0].textContent.trim();
                    if (!name || name === 'RESERVED') {
                        continue;
                    }
                    const entry = { name, version };
                    const category = categories(name);
                    if (category) {
                        entry.category = category;
                    }
                    const argumentsElements = operator.getElementsByTagName('arguments');
                    if (argumentsElements.length > 0) {
                        const inputs = [];
                        const outputs = [];
                        const attributes = [];
                        for (const arg of argumentsElements[0].getElementsByTagName('argument')) {
                            const argName = arg.getAttribute('category');
                            const argLabel = arg.getAttribute('name');
                            const descElements = arg.getElementsByTagName('description');
                            const description = descElements.length > 0 ? descElements[0].textContent.trim().replace(/\s+/g, ' ') : '';
                            const argType = arg.getAttribute('type');
                            const elementType = arg.getAttribute('tensor-element-type');
                            const rawType = elementType && elementType !== '-' ? elementType : argType;
                            const type = types.has(rawType) ? types.get(rawType) : rawType;
                            const item = { name: argLabel };
                            if (type) {
                                item.type = type;
                            }
                            if (description) {
                                item.description = description;
                            }
                            if (argName === 'input') {
                                inputs.push(item);
                            } else if (argName === 'output') {
                                outputs.push(item);
                            } else if (argName === 'attribute') {
                                attributes.push(item);
                            }
                        }
                        if (inputs.length > 0) {
                            entry.inputs = inputs;
                        }
                        if (outputs.length > 0) {
                            entry.outputs = outputs;
                        }
                        if (attributes.length > 0) {
                            entry.attributes = attributes;
                        }
                    }
                    entries.push(entry);
                }
            }
        }
    }
    entries.sort((a, b) => a.name.localeCompare(b.name) || a.version.localeCompare(b.version));
    let output = JSON.stringify(entries, null, 2);
    output = output.replace(/\s {8}/g, ' ');
    output = output.replace(/,\s {8}/g, ', ');
    output = output.replace(/\s {6}}/g, ' }');
    const file = path.join(dirname, '..', 'source', 'tosa-metadata.json');
    await fs.writeFile(file, output, 'utf-8');
};

await main();
