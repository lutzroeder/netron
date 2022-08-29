const path = require('path');
const flatc = require('./flatc');
const fs = require('fs');

const schema = path.join(__dirname, '..', 'third_party', 'source', 'megengine', 'src', 'serialization', 'fbs', 'schema_v2.fbs');
const file = path.join(__dirname, '..', 'source', 'megengine-metadata.json');

const input = fs.readFileSync(file, 'utf-8');
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

const root = new flatc.Root('megengine', [], [ schema ]);
const namespace = root.find('mgb.serialization.fbs.param', flatc.Namespace);

const operatorParams = namespace.children;
for (const op of operatorParams) {
    const op_key = op[ 0 ];
    const op_table = op[ 1 ];
    if (op_table instanceof flatc.Enum) {
        continue;
    }
    if (op_table && op_table.fields.size > 0) {
        if (!operators.has(op_key)) {
            const operator = { name: op_key };
            operators.set(op_key, operator);
            json.push(operator);
        }
        const operator = operators.get(op_key);
        if (category[ op_key.replace(/V(\d+)$/, '') ]) {
            operator.category = category[ op_key.replace(/V(\d+)$/, '') ];
        }
        operator.attributes = operator.attributes || [];
        for (const field of op_table.fields) {
            const field_name = field[ 0 ];
            const field_table = field[ 1 ];
            const attr_key = op_key + ':' + field_name;
            if (!attributes.has(attr_key)) {
                const attribute = { name: field_name };
                attributes.set(attr_key, attribute);
                operator.attributes.push(attribute);
            }
            const attribute = attributes.get(attr_key);
            const type = field_table.type;
            let defaultValue = field_table.defaultValue;
            if (type instanceof flatc.Enum) {
                if (!type.keys.has(defaultValue)) {
                    throw new Error("Invalid '" + type.name + "' default value '" + defaultValue + "'.");
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
fs.writeFileSync(file, output, 'utf-8');
