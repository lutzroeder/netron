
from __future__ import unicode_literals
from __future__ import print_function

import io
import json
import logging
import pydoc
import os
import re
import sys

def get_support_level(dir):
    dir = dir.replace('\\', '/')
    if 'caffe2/caffe2/operators' in dir:
        return 'core'
    if 'contrib' in dir.split('/'):
        return 'contribution'
    if 'experiments' in dir.split('/'):
        return 'experimental'
    return 'default'

def update_argument_type(type):
    if type == 'int' or type == 'int64_t':
        return 'int64'
    if type == 'int32_t':
        return 'int32'
    elif type == '[int]' or type == 'int[]':
        return 'int64[]'
    elif type == 'float':
        return 'float32'
    elif type == 'string':
        return 'string'
    elif type == 'List(string)':
        return 'string[]'
    elif type == 'bool':
        return 'boolean'
    raise Exception('Unknown argument type ' + str(type))

def update_argument_default(value, type):
    if type == 'int64':
        return int(value)
    elif type == 'float32':
        return float(value.rstrip('~'))
    elif type == 'boolean':
        if value == 'True':
            return True
        if value == 'False':
            return False
    elif type == 'string':
        return value.strip('\"')
    raise Exception('Unknown argument type ' + str(type))

def update_argument(schema, arg):
    if not 'attributes' in schema:
        schema['attributes'] = []
    attribute = None
    for current_attribute in schema['attributes']:
        if 'name' in current_attribute and current_attribute['name'] == arg.name:
            attribute = current_attribute
            break
    if not attribute:
        attribute = {}
        attribute['name'] = arg.name
        schema['attributes'].append(attribute)
    description = arg.description.strip()
    if description.startswith('*('):
        index = description.find(')*')
        properties = []
        if index != -1:
            properties = description[2:index].split(';')
            description = description[index+2:].lstrip()
        else:
            index = description.index(')')
            properties = description[2:index].split(';')
            description = description[index+1:].lstrip()
        if len(properties) == 1 and properties[0].find(',') != -1:
            properties = properties[0].split(',')
        for property in properties:
            parts = property.split(':')
            name = parts[0].strip()
            if name == 'type':
                type = parts[1].strip()
                if type == 'primitive' or type == 'int | Tuple(int)' or type == '[]' or type == 'TensorProto_DataType' or type == 'Tuple(int)':
                    continue
                attribute['type'] = update_argument_type(type)
            elif name == 'default':
                if 'type' in attribute:
                    type = attribute['type']
                    default = parts[1].strip()
                    if default == '2, possible values':
                        default = '2'
                    if type == 'float32' and default == '\'NCHW\'':
                        continue
                    if type == 'int64[]':
                        continue
                    attribute['default'] = update_argument_default(default, type)
            elif name == 'optional':
                attribute['option'] = 'optional'
            elif name == 'must be > 1.0' or name == 'default=\'NCHW\'' or name == 'type depends on dtype' or name == 'Required=True':
                continue
            elif name == 'List(string)':
                attribute['type'] = 'string[]'
            else:
                raise Exception('Unknown property ' + str(parts[0].strip()))
    attribute['description'] = description
    if not arg.required:
        attribute['option'] = 'optional'
    return

def update_input(schema, input_desc):
    input_name = input_desc[0]
    description = input_desc[1]
    if not 'inputs' in schema:
        schema['inputs'] = []
    input_arg = None
    for current_input in schema['inputs']:
        if 'name' in current_input and current_input['name'] == input_name:
            input_arg = current_input
            break
    if not input_arg:
        input_arg = {}
        input_arg['name'] = input_name
        schema['inputs'].append(input_arg)
    input_arg['description'] = description
    if len(input_desc) > 2:
        return

def update_output(operator_name, schema, output_desc):
    output_name = output_desc[0]
    description = output_desc[1]
    if not 'outputs' in schema:
        schema['outputs'] = []
    output_arg = None
    for current_output in schema['outputs']:
        if 'name' in current_output and current_output['name'] == output_name:
            output_arg = current_output
            break
    if not output_arg:
        output_arg = {}
        output_arg['name'] = output_name
        schema['outputs'].append(output_arg)
    output_arg['description'] = description
    if len(output_desc) > 2:
        return

class Caffe2Filter(logging.Filter):
    def filter(self, record):
        return record.getMessage().startswith('WARNING:root:This caffe2 python run does not have GPU support.')

def metadata():

    logging.getLogger('').addFilter(Caffe2Filter())

    import caffe2.python.core

    json_file = os.path.join(os.path.dirname(__file__), '../source/caffe2-metadata.json')
    json_data = open(json_file).read()
    json_root = json.loads(json_data)

    schema_map = {}

    for entry in json_root:
        operator_name = entry['name']
        schema = entry['schema']
        schema_map[operator_name] = schema

    for operator_name in caffe2.python.core._GetRegisteredOperators():
        op_schema = caffe2.python.workspace.C.OpSchema.get(operator_name)
        if op_schema:
            if operator_name == 'Crash':
                continue
            if operator_name in schema_map:
                schema = schema_map[operator_name]
            else:
                schema = {}
                entry = { 'name': operator_name, 'schema': schema }
                schema_map[operator_name] = entry
                json_root.append(entry)
            schema['description'] = op_schema.doc
            for arg in op_schema.args:
                update_argument(schema, arg)
            for input_desc in op_schema.input_desc:
                update_input(schema, input_desc)
            for output_desc in op_schema.output_desc:
                update_output(operator_name, schema, output_desc)
            schema['support_level'] = get_support_level(os.path.dirname(op_schema.file))

    with io.open(json_file, 'w', newline='') as fout:
        json_data = json.dumps(json_root, sort_keys=True, indent=2)
        for line in json_data.splitlines():
            line = line.rstrip()
            if sys.version_info[0] < 3:
                line = unicode(line)
            fout.write(line)
            fout.write('\n')

if __name__ == '__main__':
    command_table = { 'metadata': metadata }
    command = sys.argv[1];
    command_table[command]()
