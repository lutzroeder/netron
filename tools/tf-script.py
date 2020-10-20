
import io
import json
import os
import sys

from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.framework import types_pb2
from google.protobuf import text_format

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def metadata():
    categories = {
        'Assign': 'Control',
        'AvgPool': 'Pool',
        'BatchNormWithGlobalNormalization': 'Normalization',
        'BiasAdd': 'Layer',
        'ConcatV2': 'Tensor',
        'Const': 'Constant',
        'Conv2D': 'Layer',
        'DepthwiseConv2dNative': 'Layer',
        'Dequantize': 'Tensor',
        'Elu': 'Activation',
        'FusedBatchNorm': 'Normalization',
        'FusedBatchNormV2': 'Normalization',
        'FusedBatchNormV3': 'Normalization',
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
        'Squeeze': 'Shape',
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
        for c in text:
            if (c == '\n'):
                result += '\\n'
            elif (c == '\r'):
                result += "\\r"
            elif (c == '\t'):
                result += "\\t"
            elif (c == '\"'):
                result += "\\\""
            elif (c == '\''):
                result += "\\'"
            elif (c == '\\'):
                result += "\\\\"
            else:
                result += c
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
            if end == None:
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
            api_defs = api_def_pb2.ApiDefs()
            filename = folder + '/' + filename
            with open(filename) as handle:
                multiline_pbtxt = handle.read()
                pbtxt = pbtxt_from_multiline(multiline_pbtxt)
                text_format.Merge(pbtxt, api_defs)
            for api_def in api_defs.op:
                api_def_map[api_def.graph_op_name] = api_def
        return api_def_map

    def convert_type(type):
        return { 'type': 'type', 'value': type }

    def convert_tensor(tensor):
        return { 'type': 'tensor', 'value': '?' }

    def convert_shape(shape):
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

    def convert_attr_type(type):
        if type in attr_type_table:
            return attr_type_table[type]
        print(type)
        return type

    def convert_attr_value(attr_value):
        if attr_value.HasField('list'):
            list = []
            attr_value_list = attr_value.list
            if len(attr_value_list.s) > 0:
                for s in attr_value_list.s:
                    list.append(s.decode('utf8'))
            if len(attr_value_list.i) > 0:
                for i in attr_value_list.i:
                    list.append(i)
            if len(attr_value_list.f) > 0:
                for f in attr_value_list.f:
                    list.append(convert_number(f))
            if len(attr_value_list.type) > 0:
                for type in attr_value_list.type:
                    list.append(convert_type(type))
            if len(list) == 0:
                for _, value in attr_value_list.ListFields():
                    if len(value) > 0:
                        raise Exception()
            return list
        if attr_value.HasField('s'):
            return attr_value.s.decode('utf8')
        if attr_value.HasField('i'):
            return attr_value.i
        if attr_value.HasField('f'):
            return convert_number(attr_value.f)
        if attr_value.HasField('b'):
            return attr_value.b
        if attr_value.HasField('type'):
            return convert_type(attr_value.type)
        if attr_value.HasField('tensor'):
            return convert_tensor(attr_value.tensor)
        if attr_value.HasField('shape'):
            return convert_shape(attr_value.shape)
        raise Exception()

    _TYPE_TO_STRING = {
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
        if data_type in _TYPE_TO_STRING:
            return _TYPE_TO_STRING[data_type]
        raise Exception()

    def format_attribute_value(value):
        if type(value) is dict and 'type' in value and 'value' in value and value['type'] == 'type':
            return format_data_type(value['value'])
        if type(value) is str:
            return value
        if value == True:
            return 'true'
        if value == False:
            return 'false'
        raise Exception()

    tensorflow_repo_dir = os.path.join(os.path.dirname(__file__), '../third_party/source/tf')
    api_def_map = read_api_def_map(os.path.join(tensorflow_repo_dir, 'tensorflow/core/api_def/base_api'))
    input_file = os.path.join(tensorflow_repo_dir, 'tensorflow/core/ops/ops.pbtxt')
    ops_list = op_def_pb2.OpList()
    with open(input_file) as input_handle:
        text_format.Merge(input_handle.read(), ops_list)

    json_root = []

    for op in ops_list.op:
        # print(op.name)
        json_schema = {}
        if op.name in categories:
            json_schema['category'] = categories[op.name]
        api_def = api_def_pb2.ApiDef()
        if op.name in api_def_map:
            api_def = api_def_map[op.name]
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
        for attr in op.attr:
            if not 'attributes' in json_schema:
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
                description = json_attribute['description'] + ' ' if 'description' in json_attribute else ''
                description = description + 'Must be one of the following: ' + ', '.join(list(map(lambda x: "`" + format_attribute_value(x) + "`", allowed_values))) + '.'
                json_attribute['description'] = description
            if attr.HasField('default_value'):
                default_value = convert_attr_value(attr.default_value)
                json_attribute['default'] = default_value
            json_schema['attributes'].append(json_attribute)
        for input_arg in op.input_arg:
            if not 'inputs' in json_schema:
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
        for output_arg in op.output_arg:
            if not 'outputs' in json_schema:
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
        json_root.append({
            'name': op.name,
            'schema': json_schema 
        })

    json_file = os.path.join(os.path.dirname(__file__), '../source/tf-metadata.json')
    with io.open(json_file, 'w', newline='') as fout:
        json_data = json.dumps(json_root, sort_keys=True, indent=2)
        for line in json_data.splitlines():
            line = line.rstrip()
            fout.write(line)
            fout.write('\n')

if __name__ == '__main__':
    command_table = { 'metadata': metadata }
    command = sys.argv[1]
    command_table[command]()