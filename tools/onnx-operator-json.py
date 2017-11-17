#!/usr/bin/python

from __future__ import unicode_literals

import json
import io
import sys

from onnx import defs
from onnx.defs import OpSchema
from onnx.backend.test.case.node import collect_snippets

SNIPPETS = collect_snippets()

def generate_json_attr_type(type):
    assert isinstance(type, OpSchema.AttrType)
    s = str(type)
    s = s[s.rfind('.')+1:].lower()
    if s[-1] == 's':
        s = 'list of ' + s
    return s

def generate_json_support_level_name(support_level):
    assert isinstance(support_level, OpSchema.SupportType)
    s = str(support_level)
    return s[s.rfind('.')+1:].lower()

def generate_json_types(types):
    r = []
    for type in types:
        r.append(type)
    r = sorted(r)
    return r

def generate_json(sorted_ops, file):
    json_root = []
    for _, op_type, schema in sorted_ops:
        json_schema = {}
        json_schema['support_level'] = generate_json_support_level_name(schema.support_level)
        if schema.doc:
            json_schema['doc'] = schema.doc.lstrip();
        if schema.inputs:
            json_schema['inputs'] = []
            for input in schema.inputs:
                json_schema['inputs'].append({ 
                    'name': input.name, 
                    'description': input.description,
                    'optional': input.optional,
                    'typeStr': input.typeStr,
                    'types': generate_json_types(input.types) })
        json_schema['min_input'] = schema.min_input;
        json_schema['max_input'] = schema.max_input;
        if schema.outputs:
            json_schema['outputs'] = []
            for output in schema.outputs:
                json_schema['outputs'].append({ 
                    'name': output.name, 
                    'description': output.description,
                    'optional': output.optional,
                    'typeStr': output.typeStr,
                    'types': generate_json_types(output.types) })
        json_schema['min_output'] = schema.min_output;
        json_schema['max_output'] = schema.max_output;
        if schema.attributes:
            json_schema['attributes'] = []
            for _, attribute in sorted(schema.attributes.items()):
                json_schema['attributes'].append({
                    'name' : attribute.name,
                    'description': attribute.description,
                    'type': generate_json_attr_type(attribute.type),
                    'required': attribute.required })
        if schema.type_constraints:
            json_schema["type_constraints"] = []
            for type_constraint in schema.type_constraints:
                json_schema['type_constraints'].append({
                    'description': type_constraint.description,
                    'type_param_str': type_constraint.type_param_str,
                    'allowed_type_strs': type_constraint.allowed_type_strs
                })
        if op_type in SNIPPETS:
            json_schema['snippets'] = []
            for summary, code in sorted(SNIPPETS[op_type]):
                json_schema['snippets'].append({
                    'summary': summary,
                    'code': code
                })
        json_root.append({
            "op_type": op_type,
            "schema": json_schema })
    with io.open(file, 'w', newline='') as fout:
        json_root = json.dumps(json_root, sort_keys=True, indent=2)
        for line in json_root.splitlines():
            line = line.rstrip()
            if sys.version_info[0] < 3:
                line = unicode(line)
            fout.write(line)
            fout.write('\n')

if __name__ == '__main__':
    sorted_ops = sorted(
        (int(schema.support_level), op_type, schema)
        for (op_type, schema) in defs.get_all_schemas().items())
    generate_json(sorted_ops, '../src/onnx-operator.json')
