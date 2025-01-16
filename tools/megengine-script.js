
import * as flatc from './flatc.js';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as url from 'url';

const main = async () => {
    const dirname = path.dirname(url.fileURLToPath(import.meta.url));
    const schema = path.join(dirname, '..', 'third_party', 'source', 'megengine', 'src', 'serialization', 'fbs', 'schema_v2.fbs');
    const file = path.join(dirname, '..', 'source', 'megengine-metadata.json');
    const input = await fs.readFile(file, 'utf-8');
    const json = JSON.parse(input);
    const category = {
        Host2DeviceCopy: 'Data',
        Dimshuffle: 'Shape',
        Flip: 'Shape',
        Images2Neibs: 'Shape',
        Reshape: 'Shape',
        Concat: 'Tensor',
        GetVarShape: 'Shape',
        Subtensor: 'Tensor',
        Padding: 'Layer',
        AdaptivePooling: 'Activation',
        ConvPooling: 'Pool',
        TQT: 'Quantization',
        LSQ: 'Quantization',
        Pooling: 'Pool',
        PoolingForward: 'Pool',
        AdaptivePoolingForward: 'Pool',
        SlidingWindowTranspose: 'Transform',
        LRN: 'Normalization',
        BatchNormForward: 'Normalization',
        BN: 'Normalization',
        LayerNorm: 'Normalization',
        Convolution: 'Layer',
        ConvolutionForward: 'Layer',
        Convolution3D: 'Layer',
        SeparableConv: 'Layer',
        SeparableConv3D: 'Layer',
        ConvBiasForward: 'Layer',
        ConvBias: 'Layer',
        Conv3DBias: 'Layer',
        Dropout: 'Dropout',
        Softmax: 'Activation',
        RNN: 'Layer',
        RNNCell: 'Layer',
        LSTM: 'Layer'
    };
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
    const root = new flatc.Root('megengine');
    await root.load([], [schema]);
    const namespace = root.find('mgb.serialization.fbs.param', flatc.Namespace);
    const operatorParams = namespace.children;
    for (const [name, op] of operatorParams) {
        if (op instanceof flatc.Enum) {
            continue;
        }
        if (op && op.fields.size > 0) {
            if (!operators.has(name)) {
                const operator = { name };
                operators.set(name, operator);
                json.push(operator);
            }
            const operator = operators.get(name);
            const k = name.replace(/V\d+$/, '');
            if (category[k]) {
                operator.category = category[k];
            }
            operator.attributes = operator.attributes || [];
            for (const [field_name, field] of op.fields) {
                const attr_key = `${name}:${field_name}`;
                if (!attributes.has(attr_key)) {
                    const attribute = { name: field_name };
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
                attribute.type = type.name + (field.repeated ? '[]' : '');
                attribute.default = defaultValue;
            }
        }
    }
    // json.sort((a, b) => a.name.localeCompare(b.name))

    let output = JSON.stringify(json, null, 2);
    output = output.replace(/\s {8}/g, ' ');
    output = output.replace(/,\s {8}/g, ', ');
    output = output.replace(/\s {6}}/g, ' }');
    await fs.writeFile(file, output, 'utf-8');
};

await main();