#!/usr/bin/env python

from __future__ import unicode_literals

import json
import io
import sys
import os

from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from google.protobuf import text_format

categories = {
    'Const': 'Constant',
    'Conv2D': 'Layer',
    'BiasAdd': 'Layer',
    'DepthwiseConv2dNative': 'Layer',
    'Relu': 'Activation',
    'Relu6': 'Activation',
    'Softmax': 'Activation',
    'Sigmoid': 'Activation',
    'LRN': 'Normalization',
    'MaxPool': 'Pool',
    'MaxPoolV2': 'Pool',
    'AvgPool': 'Pool',
    'Reshape': 'Shape',
    'Squeeze': 'Shape',
    'ConcatV2': 'Tensor',
    'Split': 'Tensor',
    'Dequantize': 'Tensor',
    'Identity': 'Control',
    'Variable': 'Control',
    'VariableV2': 'Control',
    'Assign': 'Control',
    'BatchNormWithGlobalNormalization': 'Normalization',
    'FusedBatchNorm': 'Normalization',
    # 'VariableV2':
    # 'Assign':
    # 'BiasAdd':
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

api_def_map = read_api_def_map('../third_party/tensorflow/tensorflow/core/api_def/base_api')

input_file = '../third_party/tensorflow/tensorflow/core/ops/ops.pbtxt';

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
        if attr.type:
            json_attribute['type'] = attr.type
        if attr.name in api_def_attr_map:
            api_def_attr = api_def_attr_map[attr.name]
            if api_def_attr.description:
                json_attribute['description'] = api_def_attr.description
        if attr.has_minimum:
            json_attribute['minimum'] = attr.minimum
        if attr.HasField('allowed_values'):
            json_attribute['allowedValues'] = convert_attr_value(attr.allowed_values)
        if attr.HasField('default_value'):
            json_attribute['defaultValue'] = convert_attr_value(attr.default_value)
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

json_file = '../src/tf-metadata.json'
with io.open(json_file, 'w', newline='') as fout:
    json_data = json.dumps(json_root, sort_keys=True, indent=2)
    for line in json_data.splitlines():
        line = line.rstrip()
        if sys.version_info[0] < 3:
            line = unicode(line)
        fout.write(line)
        fout.write('\n')
