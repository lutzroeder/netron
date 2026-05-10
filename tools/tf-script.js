
import * as fs from 'fs/promises';
import * as path from 'path';
import * as protobuf from '../source/protobuf.js';
import * as url from 'url';
import { tensorflow } from '../source/tf-proto.js';

const decoder = new TextDecoder('utf-8');

const has = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);

const dataTypes = new Map([
    [tensorflow.DataType.DT_HALF, 'float16'],
    [tensorflow.DataType.DT_FLOAT, 'float32'],
    [tensorflow.DataType.DT_DOUBLE, 'float64'],
    [tensorflow.DataType.DT_INT32, 'int32'],
    [tensorflow.DataType.DT_UINT8, 'uint8'],
    [tensorflow.DataType.DT_UINT16, 'uint16'],
    [tensorflow.DataType.DT_UINT32, 'uint32'],
    [tensorflow.DataType.DT_UINT64, 'uint64'],
    [tensorflow.DataType.DT_INT16, 'int16'],
    [tensorflow.DataType.DT_INT8, 'int8'],
    [tensorflow.DataType.DT_STRING, 'string'],
    [tensorflow.DataType.DT_COMPLEX64, 'complex64'],
    [tensorflow.DataType.DT_COMPLEX128, 'complex128'],
    [tensorflow.DataType.DT_INT64, 'int64'],
    [tensorflow.DataType.DT_BOOL, 'bool'],
    [tensorflow.DataType.DT_QINT8, 'qint8'],
    [tensorflow.DataType.DT_QUINT8, 'quint8'],
    [tensorflow.DataType.DT_QINT16, 'qint16'],
    [tensorflow.DataType.DT_QUINT16, 'quint16'],
    [tensorflow.DataType.DT_QINT32, 'qint32'],
    [tensorflow.DataType.DT_BFLOAT16, 'bfloat16'],
    [tensorflow.DataType.DT_RESOURCE, 'resource'],
    [tensorflow.DataType.DT_VARIANT, 'variant'],
    [tensorflow.DataType.DT_HALF_REF, 'float16_ref'],
    [tensorflow.DataType.DT_FLOAT_REF, 'float32_ref'],
    [tensorflow.DataType.DT_DOUBLE_REF, 'float64_ref'],
    [tensorflow.DataType.DT_INT32_REF, 'int32_ref'],
    [tensorflow.DataType.DT_UINT32_REF, 'uint32_ref'],
    [tensorflow.DataType.DT_UINT8_REF, 'uint8_ref'],
    [tensorflow.DataType.DT_UINT16_REF, 'uint16_ref'],
    [tensorflow.DataType.DT_INT16_REF, 'int16_ref'],
    [tensorflow.DataType.DT_INT8_REF, 'int8_ref'],
    [tensorflow.DataType.DT_STRING_REF, 'string_ref'],
    [tensorflow.DataType.DT_COMPLEX64_REF, 'complex64_ref'],
    [tensorflow.DataType.DT_COMPLEX128_REF, 'complex128_ref'],
    [tensorflow.DataType.DT_INT64_REF, 'int64_ref'],
    [tensorflow.DataType.DT_UINT64_REF, 'uint64_ref'],
    [tensorflow.DataType.DT_BOOL_REF, 'bool_ref'],
    [tensorflow.DataType.DT_QINT8_REF, 'qint8_ref'],
    [tensorflow.DataType.DT_QUINT8_REF, 'quint8_ref'],
    [tensorflow.DataType.DT_QINT16_REF, 'qint16_ref'],
    [tensorflow.DataType.DT_QUINT16_REF, 'quint16_ref'],
    [tensorflow.DataType.DT_QINT32_REF, 'qint32_ref'],
    [tensorflow.DataType.DT_BFLOAT16_REF, 'bfloat16_ref'],
    [tensorflow.DataType.DT_RESOURCE_REF, 'resource_ref'],
    [tensorflow.DataType.DT_VARIANT_REF, 'variant_ref']
]);

const attributeTypes = new Map([
    ['type', 'type'], ['list(type)', 'type[]'],
    ['bool', 'boolean'],
    ['int', 'int64'], ['list(int)', 'int64[]'],
    ['float', 'float32'], ['list(float)', 'float32[]'],
    ['string', 'string'], ['list(string)', 'string[]'],
    ['shape', 'shape'], ['list(shape)', 'shape[]'],
    ['tensor', 'tensor'],
    ['func', 'function'], ['list(func)', 'function[]']
]);

const categories = new Map([
    ['Assign', 'Control'],
    ['AvgPool', 'Pool'],
    ['BatchNormWithGlobalNormalization', 'Normalization'],
    ['BiasAdd', 'Layer'],
    ['Concat', 'Tensor'],
    ['ConcatV2', 'Tensor'],
    ['Const', 'Constant'],
    ['Conv2D', 'Layer'],
    ['DepthwiseConv2dNative', 'Layer'],
    ['Dequantize', 'Quantization'],
    ['Elu', 'Activation'],
    ['FusedBatchNorm', 'Normalization'],
    ['FusedBatchNormV2', 'Normalization'],
    ['FusedBatchNormV3', 'Normalization'],
    ['Gather', 'Transform'],
    ['Identity', 'Control'],
    ['LeakyRelu', 'Activation'],
    ['LRN', 'Normalization'],
    ['LSTMBlockCell', 'Layer'],
    ['MaxPool', 'Pool'],
    ['MaxPoolV2', 'Pool'],
    ['MaxPoolWithArgmax', 'Pool'],
    ['Pad', 'Tensor'],
    ['QuantizeAndDequantize', 'Quantization'],
    ['QuantizeAndDequantizeV2', 'Quantization'],
    ['QuantizeAndDequantizeV3', 'Quantization'],
    ['QuantizeAndDequantizeV4', 'Quantization'],
    ['QuantizeAndDequantizeV4Grad', 'Quantization'],
    ['QuantizeDownAndShrinkRange', 'Quantization'],
    ['QuantizeV2', 'Quantization'],
    ['Relu', 'Activation'],
    ['Relu6', 'Activation'],
    ['Reshape', 'Shape'],
    ['Sigmoid', 'Activation'],
    ['Slice', 'Tensor'],
    ['Softmax', 'Activation'],
    ['Split', 'Tensor'],
    ['Squeeze', 'Transform'],
    ['StridedSlice', 'Tensor'],
    ['swish_f32', 'Activation'],
    ['Transpose', 'Transform'],
    ['Variable', 'Control'],
    ['VariableV2', 'Control']
]);

const convertFloat = (value) => {
    if (value === Infinity) {
        return 'NaN';
    }
    if (value === -Infinity) {
        return '-NaN';
    }
    return Math.fround(value);
};

const escapeString = (text) => {
    let result = '';
    for (const c of text) {
        switch (c) {
            case '\n': result += '\\n'; break;
            case '\r': result += '\\r'; break;
            case '\t': result += '\\t'; break;
            case '"': result += '\\"'; break;
            case "'": result += "\\'"; break;
            case '\\': result += '\\\\'; break;
            default: result += c; break;
        }
    }
    return result;
};

const heredoc = (input) => {
    const lines = input.split('\n');
    const output = [];
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const colon = line.indexOf(':');
        const marker = colon === -1 ? null : line.substring(colon + 1).replace(/^ +/, '');
        if (marker === null || !marker.startsWith('<<')) {
            output.push(line);
            continue;
        }
        const terminator = marker.substring(2);
        const content = [];
        let trailing = '';
        while (++i < lines.length) {
            const inner = lines[i];
            if (inner.startsWith(terminator)) {
                trailing = inner.substring(terminator.length);
                break;
            }
            content.push(inner);
        }
        output.push(`${line.substring(0, colon + 1)}"${escapeString(content.join('\n'))}"${trailing}`);
    }
    return output.join('\n');
};

const readOpList = async (file) => {
    const content = await fs.readFile(file, 'utf-8');
    const encoder = new TextEncoder();
    const reader = new protobuf.TextReader(encoder.encode(content));
    return tensorflow.OpList.decodeText(reader);
};

const readApiDefs = async (folder) => {
    const dirs = await fs.readdir(folder);
    const files = dirs.filter((name) => name.endsWith('.pbtxt'));
    const encoder = new TextEncoder();
    const defs = new Map();
    for (const name of files.sort()) {
        const file = path.join(folder, name);
        // eslint-disable-next-line no-await-in-loop
        const content = await fs.readFile(file, 'utf-8');
        const text = heredoc(content);
        const buffer = encoder.encode(text);
        const reader = protobuf.TextReader.open(buffer);
        const apiDefs = tensorflow.ApiDefs.decodeText(reader);
        for (const op of apiDefs.op) {
            defs.set(op.graph_op_name, op);
        }
    }
    return defs;
};

const convertAttrList = (attrValue) => {
    const result = [];
    const list = attrValue.list;
    for (const value of list.s) {
        result.push(decoder.decode(value));
    }
    for (const value of list.i) {
        result.push(typeof value === 'bigint' ? Number(value) : value);
    }
    for (const value of list.f) {
        result.push(convertFloat(value));
    }
    for (const value of list.type) {
        result.push({ type: 'type', value });
    }
    if (result.length === 0 && (list.b.length > 0 || list.shape.length > 0 ||
        list.tensor.length > 0 || list.func.length > 0)) {
        throw new Error('Unsupported list value.');
    }
    return result;
};

const convertAttrValue = (attrValue) => {
    if (has(attrValue, 'list')) {
        return convertAttrList(attrValue);
    }
    if (has(attrValue, 's')) {
        return decoder.decode(attrValue.s);
    }
    if (has(attrValue, 'i')) {
        return typeof attrValue.i === 'bigint' ? Number(attrValue.i) : attrValue.i;
    }
    if (has(attrValue, 'f')) {
        return convertFloat(attrValue.f);
    }
    if (has(attrValue, 'b')) {
        return attrValue.b;
    }
    if (has(attrValue, 'type')) {
        return { type: 'type', value: attrValue.type };
    }
    if (has(attrValue, 'tensor')) {
        return { type: 'tensor', value: '?' };
    }
    if (has(attrValue, 'shape')) {
        return { type: 'shape', value: '?' };
    }
    throw new Error('Unsupported attribute value.');
};

const formatAttributeValue = (value) => {
    if (value && typeof value === 'object' && value.type === 'type') {
        if (!dataTypes.has(value.value)) {
            throw new Error(`Unknown data type '${value.value}'.`);
        }
        return dataTypes.get(value.value);
    }
    if (typeof value === 'string') {
        return value;
    }
    if (value === true || value === false) {
        return value.toString();
    }
    throw new Error('Unsupported attribute value.');
};

const buildSchema = (operator, apiDef) => {
    const schema = { name: operator.name };
    if (categories.has(operator.name)) {
        schema.category = categories.get(operator.name);
    }
    if (apiDef.summary) {
        schema.summary = apiDef.summary;
    }
    if (apiDef.description) {
        schema.description = apiDef.description;
    }
    const apiDefAttr = new Map(apiDef.attr.map((attr) => [attr.name, attr]));
    for (const attr of operator.attr) {
        if (!attributeTypes.has(attr.type)) {
            throw new Error(`Unknown attribute type '${attr.type}'.`);
        }
        const json = { name: attr.name, type: attributeTypes.get(attr.type) };
        const description = apiDefAttr.get(attr.name)?.description;
        if (description) {
            json.description = description;
        }
        if (attr.has_minimum) {
            json.minimum = Number(attr.minimum);
        }
        if (attr.allowed_values !== null) {
            const allowed = convertAttrValue(attr.allowed_values).map((v) => `\`${formatAttributeValue(v)}\``).join(', ');
            const prefix = json.description ? `${json.description} ` : '';
            json.description = `${prefix}Must be one of the following: ${allowed}.`;
        }
        if (attr.default_value !== null) {
            json.default = convertAttrValue(attr.default_value);
        }
        schema.attributes = schema.attributes || [];
        schema.attributes.push(json);
    }
    const apiDefIn = new Map(apiDef.in_arg.map((arg) => [arg.name, arg]));
    for (const arg of operator.input_arg) {
        const json = { name: arg.name };
        const description = apiDefIn.get(arg.name)?.description;
        if (description) {
            json.description = description;
        }
        if (arg.number_attr) {
            json.numberAttr = arg.number_attr;
        }
        if (arg.type) {
            json.type = arg.type;
        }
        if (arg.type_attr) {
            json.typeAttr = arg.type_attr;
        }
        if (arg.type_list_attr) {
            json.typeListAttr = arg.type_list_attr;
        }
        if (arg.is_ref) {
            json.isRef = true;
        }
        schema.inputs = schema.inputs || [];
        schema.inputs.push(json);
    }
    const apiDefOut = new Map(apiDef.out_arg.map((arg) => [arg.name, arg]));
    for (const arg of operator.output_arg) {
        const json = { name: arg.name };
        const description = apiDefOut.get(arg.name)?.description;
        if (description) {
            json.description = description;
        }
        if (arg.number_attr) {
            json.numberAttr = arg.number_attr;
        }
        if (arg.type) {
            json.type = arg.type;
        } else if (arg.type_attr) {
            json.typeAttr = arg.type_attr;
        } else if (arg.type_list_attr) {
            json.typeListAttr = arg.type_list_attr;
        }
        if (arg.is_ref) {
            json.isRef = true;
        }
        schema.outputs = schema.outputs || [];
        schema.outputs.push(json);
    }
    return schema;
};

const main = async () => {
    const root = path.dirname(path.dirname(url.fileURLToPath(import.meta.url)));
    const dir = path.join(root, 'third_party', 'source', 'tensorflow', 'tensorflow', 'core');
    const defs = await readApiDefs(path.join(dir, 'api_def', 'base_api'));
    const ops = await readOpList(path.join(dir, 'ops', 'ops.pbtxt'));
    const schemas = [];
    for (const op of ops.op) {
        const apiDef = defs.get(op.name) || new tensorflow.ApiDef();
        const schema = buildSchema(op, apiDef);
        schemas.push(schema);
    }
    const file = path.join(root, 'source', 'tf-metadata.json');
    const content = JSON.stringify(schemas, null, 2);
    await fs.writeFile(file, content, 'utf-8');
};

await main();
