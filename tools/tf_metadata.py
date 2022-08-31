''' TensorFlow Metadata Script '''

import json
import os
import google.protobuf.text_format
from tensorflow.core.framework import api_def_pb2 # pylint: disable=import-error
from tensorflow.core.framework import op_def_pb2 # pylint: disable=import-error
from tensorflow.core.framework import types_pb2 # pylint: disable=import-error

def _read(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

def _write(path, content):
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)

def _find_multiline(line, colon):
    if colon == -1:
        return None
    line = line[colon+1:]
    while line.startswith(' '):
        line = line[1:]
    if line.startswith('<<'):
        line = line[2:]
        return line
    return None

def _str_escape(text):
    result = ''
    for value in text:
        if value == '\n':
            result += '\\n'
        elif value == '\r':
            result += "\\r"
        elif value == '\t':
            result += "\\t"
        elif value == '\"':
            result += "\\\""
        elif value == '\'':
            result += "\\'"
        elif value == '\\':
            result += "\\\\"
        else:
            result += value
    return result

def _pbtxt_from_multiline(multiline_pbtxt):
    pbtxt = ''
    while len(multiline_pbtxt) > 0:
        index = multiline_pbtxt.find('\n')
        if index == -1:
            pbtxt = pbtxt + multiline_pbtxt
            multiline_pbtxt = ''
            break
        line = multiline_pbtxt[0:index]
        multiline_pbtxt = multiline_pbtxt[index+1:]
        colon = line.find(':')
        end = _find_multiline(line, colon)
        if end is None:
            pbtxt = pbtxt + line + '\n'
            continue
        pbtxt = pbtxt + line[0:colon+1]
        unescaped = ''
        newline = False
        line = ''
        while len(multiline_pbtxt) > 0:
            index = multiline_pbtxt.find('\n')
            line = multiline_pbtxt[0:index]
            multiline_pbtxt = multiline_pbtxt[index+1:]
            if line.startswith(end):
                line = line[len(end):]
                break
            if newline:
                unescaped = unescaped + '\n'
            newline = True
            unescaped = unescaped + line
            line = ''
        pbtxt = pbtxt + '\"' + _str_escape(unescaped) + '\"' + line + '\n'
    return pbtxt

def _read_op_list(file):
    op_list = op_def_pb2.OpList()
    content = _read(file)
    google.protobuf.text_format.Merge(content, op_list)
    return op_list

def _read_api_def_map(folder):
    api_def_map = {}
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.pbtxt'):
            api_defs = api_def_pb2.ApiDefs()
            filename = folder + '/' + filename
            with open(filename, 'r', encoding='utf-8') as file:
                multiline_pbtxt = file.read()
                pbtxt = _pbtxt_from_multiline(multiline_pbtxt)
                google.protobuf.text_format.Merge(pbtxt, api_defs)
            for api_def in api_defs.op:
                api_def_map[api_def.graph_op_name] = api_def
    return api_def_map

def _convert_type(value):
    return { 'type': 'type', 'value': value }

def _convert_tensor(tensor): # pylint: disable=unused-argument
    return { 'type': 'tensor', 'value': '?' }

def _convert_shape(shape): # pylint: disable=unused-argument
    return { 'type': 'shape', 'value': '?' }

def _convert_number(number):
    if number == float('inf'):
        return 'NaN'
    if number == float('-inf'):
        return '-NaN'
    return number

attr_type_table = {
    'type': 'type', 'list(type)': 'type[]',
    'bool': 'boolean',
    'int': 'int64', 'list(int)': 'int64[]',
    'float': 'float32', 'list(float)': 'float32[]',
    'string': 'string', 'list(string)': 'string[]',
    'shape': 'shape', 'list(shape)': 'shape[]',
    'tensor': 'tensor',
    'func': 'function', 'list(func)': 'function[]'
}

def _convert_attr_type(attr_type):
    if attr_type in attr_type_table:
        return attr_type_table[attr_type]
    print(attr_type)
    return attr_type

def _convert_attr_list(attr_value):
    result = []
    attr_value_list = attr_value.list
    if len(attr_value_list.s) > 0:
        for value in attr_value_list.s:
            result.append(value.decode('utf8'))
    if len(attr_value_list.i) > 0:
        for i in attr_value_list.i:
            result.append(i)
    if len(attr_value_list.f) > 0:
        for value in attr_value_list.f:
            result.append(_convert_number(value))
    if len(attr_value_list.type) > 0:
        for value in attr_value_list.type:
            result.append(_convert_type(value))
    if len(result) == 0:
        for _, value in attr_value_list.ListFields():
            if len(value) > 0:
                raise Exception()
    return result

def _convert_attr_value(attr_value):
    if attr_value.HasField('list'):
        value = _convert_attr_list(attr_value)
    elif attr_value.HasField('s'):
        value = attr_value.s.decode('utf8')
    elif attr_value.HasField('i'):
        value = attr_value.i
    elif attr_value.HasField('f'):
        value = _convert_number(attr_value.f)
    elif attr_value.HasField('b'):
        value = attr_value.b
    elif attr_value.HasField('type'):
        value = _convert_type(attr_value.type)
    elif attr_value.HasField('tensor'):
        value = _convert_tensor(attr_value.tensor)
    elif attr_value.HasField('shape'):
        value = _convert_shape(attr_value.shape)
    else:
        raise Exception()
    return value

type_to_string_map = {
    types_pb2.DataType.DT_HALF: "float16",
    types_pb2.DataType.DT_FLOAT: "float32",
    types_pb2.DataType.DT_DOUBLE: "float64",
    types_pb2.DataType.DT_INT32: "int32",
    types_pb2.DataType.DT_UINT8: "uint8",
    types_pb2.DataType.DT_UINT16: "uint16",
    types_pb2.DataType.DT_UINT32: "uint32",
    types_pb2.DataType.DT_UINT64: "uint64",
    types_pb2.DataType.DT_INT16: "int16",
    types_pb2.DataType.DT_INT8: "int8",
    types_pb2.DataType.DT_STRING: "string",
    types_pb2.DataType.DT_COMPLEX64: "complex64",
    types_pb2.DataType.DT_COMPLEX128: "complex128",
    types_pb2.DataType.DT_INT64: "int64",
    types_pb2.DataType.DT_BOOL: "bool",
    types_pb2.DataType.DT_QINT8: "qint8",
    types_pb2.DataType.DT_QUINT8: "quint8",
    types_pb2.DataType.DT_QINT16: "qint16",
    types_pb2.DataType.DT_QUINT16: "quint16",
    types_pb2.DataType.DT_QINT32: "qint32",
    types_pb2.DataType.DT_BFLOAT16: "bfloat16",
    types_pb2.DataType.DT_RESOURCE: "resource",
    types_pb2.DataType.DT_VARIANT: "variant",
    types_pb2.DataType.DT_HALF_REF: "float16_ref",
    types_pb2.DataType.DT_FLOAT_REF: "float32_ref",
    types_pb2.DataType.DT_DOUBLE_REF: "float64_ref",
    types_pb2.DataType.DT_INT32_REF: "int32_ref",
    types_pb2.DataType.DT_UINT32_REF: "uint32_ref",
    types_pb2.DataType.DT_UINT8_REF: "uint8_ref",
    types_pb2.DataType.DT_UINT16_REF: "uint16_ref",
    types_pb2.DataType.DT_INT16_REF: "int16_ref",
    types_pb2.DataType.DT_INT8_REF: "int8_ref",
    types_pb2.DataType.DT_STRING_REF: "string_ref",
    types_pb2.DataType.DT_COMPLEX64_REF: "complex64_ref",
    types_pb2.DataType.DT_COMPLEX128_REF: "complex128_ref",
    types_pb2.DataType.DT_INT64_REF: "int64_ref",
    types_pb2.DataType.DT_UINT64_REF: "uint64_ref",
    types_pb2.DataType.DT_BOOL_REF: "bool_ref",
    types_pb2.DataType.DT_QINT8_REF: "qint8_ref",
    types_pb2.DataType.DT_QUINT8_REF: "quint8_ref",
    types_pb2.DataType.DT_QINT16_REF: "qint16_ref",
    types_pb2.DataType.DT_QUINT16_REF: "quint16_ref",
    types_pb2.DataType.DT_QINT32_REF: "qint32_ref",
    types_pb2.DataType.DT_BFLOAT16_REF: "bfloat16_ref",
    types_pb2.DataType.DT_RESOURCE_REF: "resource_ref",
    types_pb2.DataType.DT_VARIANT_REF: "variant_ref",
}

def _format_data_type(data_type):
    if data_type in type_to_string_map:
        return type_to_string_map[data_type]
    raise Exception()

def _format_attribute_value(value):
    if isinstance(value, dict) and \
        'type' in value and 'value' in value and value['type'] == 'type':
        return _format_data_type(value['value'])
    if isinstance(value, str):
        return value
    if value is True:
        return 'true'
    if value is False:
        return 'false'
    raise Exception()

def _update_attributes(json_schema, operator, api_def):
    api_def_attr_map = {}
    for attr in api_def.attr:
        api_def_attr_map[attr.name] = attr
    for attr in operator.attr:
        if 'attributes' not in json_schema:
            json_schema['attributes'] = []
        json_attribute = {}
        json_attribute['name'] = attr.name
        attr_type = _convert_attr_type(attr.type)
        if attr_type:
            json_attribute['type'] = attr_type
        else:
            del json_attribute['type']
        if attr.name in api_def_attr_map:
            api_def_attr = api_def_attr_map[attr.name]
            if api_def_attr.description:
                json_attribute['description'] = api_def_attr.description
        if attr.has_minimum:
            json_attribute['minimum'] = attr.minimum
        if attr.HasField('allowed_values'):
            allowed_values = _convert_attr_value(attr.allowed_values)
            description = json_attribute['description'] + \
                ' ' if 'description' in json_attribute else ''
            allowed_values = list( \
                map(lambda x: "`" + _format_attribute_value(x) + "`", \
                allowed_values))
            description = description + \
                'Must be one of the following: ' + ', '.join(allowed_values) + '.'
            json_attribute['description'] = description
        if attr.HasField('default_value'):
            default_value = _convert_attr_value(attr.default_value)
            json_attribute['default'] = default_value
        json_schema['attributes'].append(json_attribute)

def _update_inputs(json_schema, operator, api_def):
    api_def_in_arg_map = {}
    for in_arg in api_def.in_arg:
        api_def_in_arg_map[in_arg.name] = in_arg
    for input_arg in operator.input_arg:
        if 'inputs' not in json_schema:
            json_schema['inputs'] = []
        json_input = {}
        json_input['name'] = input_arg.name
        if input_arg.name in api_def_in_arg_map:
            api_def_in_arg = api_def_in_arg_map[input_arg.name]
            if api_def_in_arg.description:
                json_input['description'] = api_def_in_arg.description
        if input_arg.number_attr:
            json_input['numberAttr'] = input_arg.number_attr
        if input_arg.type:
            json_input['type'] = input_arg.type
        if input_arg.type_attr:
            json_input['typeAttr'] = input_arg.type_attr
        if input_arg.type_list_attr:
            json_input['typeListAttr'] = input_arg.type_list_attr
        if input_arg.is_ref:
            json_input['isRef'] = True
        json_schema['inputs'].append(json_input)

def _update_outputs(json_schema, operator, api_def):
    api_def_out_arg_map = {}
    for out_arg in api_def.out_arg:
        api_def_out_arg_map[out_arg.name] = out_arg
    for output_arg in operator.output_arg:
        if 'outputs' not in json_schema:
            json_schema['outputs'] = []
        json_output = {}
        json_output['name'] = output_arg.name
        if output_arg.name in api_def_out_arg_map:
            api_def_out_arg = api_def_out_arg_map[output_arg.name]
            if api_def_out_arg.description:
                json_output['description'] = api_def_out_arg.description
        if output_arg.number_attr:
            json_output['numberAttr'] = output_arg.number_attr
        if output_arg.type:
            json_output['type'] = output_arg.type
        elif output_arg.type_attr:
            json_output['typeAttr'] = output_arg.type_attr
        elif output_arg.type_list_attr:
            json_output['typeListAttr'] = output_arg.type_list_attr
        if output_arg.is_ref:
            json_output['isRef'] = True
        json_schema['outputs'].append(json_output)

categories = {
    'Assign': 'Control',
    'AvgPool': 'Pool',
    'BatchNormWithGlobalNormalization': 'Normalization',
    'BiasAdd': 'Layer',
    'Concat': 'Tensor',
    'ConcatV2': 'Tensor',
    'Const': 'Constant',
    'Conv2D': 'Layer',
    'DepthwiseConv2dNative': 'Layer',
    'Dequantize': 'Tensor',
    'Elu': 'Activation',
    'FusedBatchNorm': 'Normalization',
    'FusedBatchNormV2': 'Normalization',
    'FusedBatchNormV3': 'Normalization',
    'Gather': 'Transform',
    'Identity': 'Control',
    'LeakyRelu': 'Activation',
    'LRN': 'Normalization',
    'LSTMBlockCell': 'Layer',
    'MaxPool': 'Pool',
    'MaxPoolV2': 'Pool',
    'MaxPoolWithArgmax': 'Pool',
    'Pad': 'Tensor',
    'Relu': 'Activation',
    'Relu6': 'Activation',
    'Reshape': 'Shape',
    'Sigmoid': 'Activation',
    'Slice': 'Tensor',
    'Softmax': 'Activation',
    'Split': 'Tensor',
    'Squeeze': 'Transform',
    'StridedSlice': 'Tensor',
    'swish_f32': 'Activation',
    'Variable': 'Control',
    'VariableV2': 'Control',
}

def _metadata():
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    core_dir = os.path.join(root_dir, 'third_party', 'source', 'tensorflow', 'tensorflow', 'core')
    api_def_map = _read_api_def_map(os.path.join(core_dir, 'api_def' , 'base_api'))
    ops_list = _read_op_list(os.path.join(core_dir, 'ops', 'ops.pbtxt'))

    json_root = []
    for operator in ops_list.op:
        json_schema = {}
        json_schema['name'] = operator.name
        if operator.name in categories:
            json_schema['category'] = categories[operator.name]
        api_def = api_def_pb2.ApiDef()
        if operator.name in api_def_map:
            api_def = api_def_map[operator.name]
        if api_def.summary:
            json_schema['summary'] = api_def.summary
        if api_def.description:
            json_schema['description'] = api_def.description
        _update_attributes(json_schema, operator, api_def)
        _update_inputs(json_schema, operator, api_def)
        _update_outputs(json_schema, operator, api_def)
        json_root.append(json_schema)
    json_file = os.path.join(root_dir, 'source', 'tf-metadata.json')
    _write(json_file, json.dumps(json_root, sort_keys=False, indent=2))

def main(): # pylint: disable=missing-function-docstring
    _metadata()

if __name__ == '__main__':
    main()
