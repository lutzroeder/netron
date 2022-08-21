''' TensorFlow Metadata Script '''

import io
import json
import os
import google.protobuf.text_format
from tensorflow.core.framework import api_def_pb2 # pylint: disable=import-error
from tensorflow.core.framework import op_def_pb2 # pylint: disable=import-error
from tensorflow.core.framework import types_pb2 # pylint: disable=import-error

def _metadata():
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

    def find_multiline(line, colon):
        if colon == -1:
            return None
        line = line[colon+1:]
        while line.startswith(' '):
            line = line[1:]
        if line.startswith('<<'):
            line = line[2:]
            return line
        return None

    def str_escape(text):
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

    def pbtxt_from_multiline(multiline_pbtxt):
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
            end = find_multiline(line, colon)
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
            pbtxt = pbtxt + '\"' + str_escape(unescaped) + '\"' + line + '\n'
        return pbtxt

    def read_api_def_map(folder):
        api_def_map = {}
        file_list = os.listdir(folder)
        file_list = sorted(file_list)
        for filename in file_list:
            if filename.endswith('.pbtxt'):
                api_defs = api_def_pb2.ApiDefs()
                filename = folder + '/' + filename
                with open(filename, 'r', encoding='utf-8') as file:
                    multiline_pbtxt = file.read()
                    pbtxt = pbtxt_from_multiline(multiline_pbtxt)
                    google.protobuf.text_format.Merge(pbtxt, api_defs)
                for api_def in api_defs.op:
                    api_def_map[api_def.graph_op_name] = api_def
        return api_def_map

    def convert_type(value):
        return { 'type': 'type', 'value': value }

    def convert_tensor(tensor): # pylint: disable=unused-argument
        return { 'type': 'tensor', 'value': '?' }

    def convert_shape(shape): # pylint: disable=unused-argument
        return { 'type': 'shape', 'value': '?' }

    def convert_number(number):
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

    def convert_attr_type(attr_type):
        if attr_type in attr_type_table:
            return attr_type_table[attr_type]
        print(attr_type)
        return attr_type

    def convert_attr_value(attr_value):
        def convert_attr_list(attr_value):
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
                    result.append(convert_number(value))
            if len(attr_value_list.type) > 0:
                for value in attr_value_list.type:
                    result.append(convert_type(value))
            if len(result) == 0:
                for _, value in attr_value_list.ListFields():
                    if len(value) > 0:
                        raise Exception()
            return result
        if attr_value.HasField('list'):
            value = convert_attr_list(attr_value)
        elif attr_value.HasField('s'):
            value = attr_value.s.decode('utf8')
        elif attr_value.HasField('i'):
            value = attr_value.i
        elif attr_value.HasField('f'):
            value = convert_number(attr_value.f)
        elif attr_value.HasField('b'):
            value = attr_value.b
        elif attr_value.HasField('type'):
            value = convert_type(attr_value.type)
        elif attr_value.HasField('tensor'):
            value = convert_tensor(attr_value.tensor)
        elif attr_value.HasField('shape'):
            value = convert_shape(attr_value.shape)
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

    def format_data_type(data_type):
        if data_type in type_to_string_map:
            return type_to_string_map[data_type]
        raise Exception()

    def format_attribute_value(value):
        if isinstance(value, dict) and \
            'type' in value and 'value' in value and value['type'] == 'type':
            return format_data_type(value['value'])
        if isinstance(value, str):
            return value
        if value is True:
            return 'true'
        if value is False:
            return 'false'
        raise Exception()

    repo_dir = os.path.join(os.path.dirname(__file__), '../third_party/source/tensorflow')
    api_def_map = read_api_def_map(os.path.join(repo_dir, 'tensorflow/core/api_def/base_api'))
    input_file = os.path.join(repo_dir, 'tensorflow/core/ops/ops.pbtxt')
    ops_list = op_def_pb2.OpList()
    with open(input_file, 'r', encoding='utf-8') as file:
        google.protobuf.text_format.Merge(file.read(), ops_list)

    json_root = []

    for operator in ops_list.op:
        # print(op.name)
        json_schema = {}
        json_schema['name'] = operator.name
        if operator.name in categories:
            json_schema['category'] = categories[operator.name]
        api_def = api_def_pb2.ApiDef()
        if operator.name in api_def_map:
            api_def = api_def_map[operator.name]
        # if op.deprecation.version != 0:
        #    print('[' + op.name + ']')
        #    print(op.deprecation.version)
        #    print(op.deprecation.explanation)
        api_def_attr_map = {}
        for attr in api_def.attr:
            api_def_attr_map[attr.name] = attr
        api_def_in_arg_map = {}
        for in_arg in api_def.in_arg:
            api_def_in_arg_map[in_arg.name] = in_arg
        api_def_out_arg_map = {}
        for out_arg in api_def.out_arg:
            api_def_out_arg_map[out_arg.name] = out_arg
        if api_def.summary:
            json_schema['summary'] = api_def.summary
        if api_def.description:
            json_schema['description'] = api_def.description
        for attr in operator.attr:
            if 'attributes' not in json_schema:
                json_schema['attributes'] = []
            json_attribute = {}
            json_attribute['name'] = attr.name
            attr_type = convert_attr_type(attr.type)
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
                allowed_values = convert_attr_value(attr.allowed_values)
                description = json_attribute['description'] + \
                    ' ' if 'description' in json_attribute else ''
                allowed_values = list( \
                    map(lambda x: "`" + format_attribute_value(x) + "`", \
                    allowed_values))
                description = description + \
                    'Must be one of the following: ' + ', '.join(allowed_values) + '.'
                json_attribute['description'] = description
            if attr.HasField('default_value'):
                default_value = convert_attr_value(attr.default_value)
                json_attribute['default'] = default_value
            json_schema['attributes'].append(json_attribute)
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
        json_root.append(json_schema)

    json_file = os.path.join(os.path.dirname(__file__), '../source/tf-metadata.json')
    with io.open(json_file, 'w', encoding='utf-8', newline='') as fout:
        json_data = json.dumps(json_root, sort_keys=False, indent=2)
        for line in json_data.splitlines():
            line = line.rstrip()
            fout.write(line)
            fout.write('\n')

def main(): # pylint: disable=missing-function-docstring
    _metadata()

if __name__ == '__main__':
    main()
