#!/usr/bin/env python

from __future__ import unicode_literals
from __future__ import print_function

import io
import json
import pydoc
import os
import re
import sys
import caffe2.python.core

json_file = '../src/caffe2-operator.json'
json_data = open(json_file).read()
json_root = json.loads(json_data)

def get_support_level(dir):
    if 'caffe2/caffe2/operators' in dir:
        return 'core'
    if 'contrib' in dir.split('/'):
        return 'contribution'
    if 'experiments' in dir.split('/'):
        return 'experimental'
    return 'default'

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
    attribute['description'] = arg.description
    if not arg.required:
        attribute['option'] = 'optional'
    return

def update_input(schema, input_desc):
    name = input_desc[0]
    description = input_desc[1]
    if not 'inputs' in schema:
        schema['inputs'] = []
    input_arg = None
    for current_input in schema['inputs']:
        if 'name' in current_input and current_input['name'] == name:
            input_arg = current_input
            break
    if not input_arg:
        input_arg = {}
        input_arg['name'] = name
        schema['inputs'].append(input_arg)
    input_arg['description'] = description
    if len(input_desc) > 2:
        return

def update_output(schema, output_desc):
    name = output_desc[0]
    description = output_desc[1]
    if not 'outputs' in schema:
        schema['outputs'] = []
    output_arg = None
    for current_output in schema['outputs']:
        if 'name' in current_output and current_output['name'] == name:
            output_arg = current_output
            break
    if not output_arg:
        output_arg = {}
        output_arg['name'] = name
        schema['outputs'].append(output_arg)
    output_arg['description'] = description
    if len(output_desc) > 2:
        return

schema_map = {}

for entry in json_root:
    name = entry['name']
    schema = entry['schema']
    schema_map[name] = schema

for name in caffe2.python.core._GetRegisteredOperators():
    op_schema = caffe2.python.workspace.C.OpSchema.get(name)
    if op_schema:
        if name in schema_map:
            schema = schema_map[name]
        else:
            schema = {}
            schema_map[name] = { 'name': name, 'schema': schema }
        schema['description'] = op_schema.doc
        for arg in op_schema.args:
            update_argument(schema, arg)
        for input_desc in op_schema.input_desc:
            update_input(schema, input_desc)
        for output_desc in op_schema.output_desc:
            update_output(schema, output_desc)
        schema['support_level'] = get_support_level(os.path.dirname(op_schema.file))

with io.open(json_file, 'w', newline='') as fout:
    json_data = json.dumps(json_root, sort_keys=True, indent=2)
    for line in json_data.splitlines():
        line = line.rstrip()
        if sys.version_info[0] < 3:
            line = unicode(line)
        fout.write(line)
        fout.write('\n')
