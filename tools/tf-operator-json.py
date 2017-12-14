#!/usr/bin/env python

from __future__ import unicode_literals

import json
import io
import sys

from tensorflow.core.framework import op_def_pb2
from google.protobuf import text_format

ops_file = '../third_party/tensorflow/tensorflow/core/ops/ops.pbtxt';
ops_list = op_def_pb2.OpList()

json_file = '../src/tf-operator.json'

with open(ops_file) as text_file:
    text = text_file.read()
    text_format.Merge(text, ops_list)

json_root = []

for schema in ops_list.op:
    json_schema = {}

    if schema.summary:
        json_schema['summary'] = schema.summary
    if schema.description:
        json_schema['description'] = schema.description
    if schema.input_arg:
        json_schema['inputs'] = []
        for input_arg in schema.input_arg:
            json_schema['inputs'].append({
                'name': input_arg.name,
                'type': input_arg.type,
                'description': input_arg.description
            })
    if schema.output_arg:
        json_schema['outputs'] = []
        for output_arg in schema.output_arg:
            json_schema['outputs'].append({
                'name': output_arg.name,
                'type': output_arg.type,
                'description': output_arg.description
            })
    if schema.attr:
        json_schema['attributes'] = []
        for attr in schema.attr:
            json_schema['attributes'].append({
                'name': attr.name,
                'type': attr.type,
                'description': attr.description
            })
    json_root.append({
        'name': schema.name,
        'schema': json_schema
    })

with io.open(json_file, 'w', newline='') as fout:
    json_root = json.dumps(json_root, sort_keys=True, indent=2)
    for line in json_root.splitlines():
        line = line.rstrip()
        if sys.version_info[0] < 3:
            line = unicode(line)
        fout.write(line)
        fout.write('\n')


