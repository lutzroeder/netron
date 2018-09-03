#!/usr/bin/env python

from __future__ import unicode_literals
from __future__ import print_function

import io
import json
import pydoc
import re
import sys

json_file = '../../src/sklearn-metadata.json'
json_data = open(json_file).read()
json_root = json.loads(json_data)

def split_docstring(docstring):
    headers = {}
    current_header = ''
    current_lines = []
    lines = docstring.split('\n')
    index = 0;
    while index < len(lines):
        if index + 1 < len(lines) and len(lines[index + 1].strip(' ')) > 0 and len(lines[index + 1].strip(' ').strip('-')) == 0:
            headers[current_header] = current_lines
            current_header = lines[index].strip(' ')
            current_lines = []
            index = index + 1
        else:
            current_lines.append(lines[index])
        index = index + 1
    headers[current_header] = current_lines
    return headers

def update_description(schema, lines):
    if len(''.join(lines).strip(' ')) > 0:
        for i in range(0, len(lines)):
            lines[i] = lines[i].lstrip(' ')
        schema['description'] = '\n'.join(lines)

def update_attribute(schema, name, description, type, option, default):
    attribute = None
    if not 'attributes' in schema:
        schema['attributes'] = []
    for current_attribute in schema['attributes']:
        if 'name' in current_attribute and current_attribute['name'] == name:
            attribute = current_attribute
            break
    if not attribute:
        attribute = {}
        attribute['name'] = name
        schema['attributes'].append(attribute)
    attribute['description'] = description
    if type:
        attribute['type'] = type
    if option:
        attribute['option'] = option
    if default:
        if type == 'float32':
            if default != "'auto'":
                attribute['default'] = float(default)
            else:
                attribute['default'] = default.strip("'").strip('"')
        elif type == 'int32':
            if default == 'None':
                attribute['default'] = None
            elif default == "'auto'" or default == '"auto"':
                attribute['default'] = default.strip("'").strip('"')
            else:
                attribute['default'] = int(default)
        elif type == 'string':
            attribute['default'] = default.strip("'").strip('"')
        elif type == 'boolean':
            if default == 'True':
                attribute['default'] = True
            elif default == 'False':
                attribute['default'] = False
            else:
                raise Exception("Unknown boolean default value '" + str(default) + "'.")
        else:
            if type:
                raise Exception("Unknown default type '" + type + "'.")
            else:
                attribute['default'] = default.strip("'")

def update_attributes(schema, lines):
    index = 0;
    while index < len(lines):
        line = lines[index]
        colon = line.find(':')
        if colon == -1:
            raise Exception("Expected ':' in parameter.")
        name = line[0:colon].strip(' ')
        line = line[colon + 1:].strip(' ')
        type = None
        if line.startswith('{'):
            end = line.find('},')
            if end == -1:
                raise Exception("Expected '}' in parameter.")
            # type = line[0:end + 1]
            line = line[end + 2:].strip(' ')
        elif line.startswith("'"):
            while line.startswith("'"):
                end = line.find("',")
                if end == -1:
                    raise Exception("Expected \' in parameter.")
                line = line[end + 2:].strip(' ')
        else:
            comma = line.find(',')
            if comma == -1:
                comma = line.find(' (')
                if comma == -1:
                    raise Exception("Expected ',' in parameter.")
            type = line[0:comma]
            line = line[comma + 1:].strip(' ')
        type_map = { 'float': 'float32', 'boolean': 'boolean', 'bool': 'boolean', 'string': 'string', 'int': 'int32' }
        if type in type_map:
            type = type_map[type]
        else:
            type = None
        # elif type == "{dict, 'balanced'}":
        #    type = 'map'
        # else:
        #    raise Exception("Unknown attribute type '" + type + "'.")
        option = None
        default = None
        while len(line.strip(' ')) > 0:
            line = line.strip(' ');
            if line.startswith('optional ') or line.startswith('optional,'):
                option = 'optional'
                line = line[9:]
            elif line.startswith('optional'):
                option = 'optional'
                line = ''
            elif line.startswith('('):
                close = line.index(')')
                if (close == -1):
                    raise Exception("Expected ')' in parameter.")
                line = line[1:close]
            elif line.endswith(' by default'):
                default = line[0:-11]
                line = ''
            elif line.startswith('default ') or line.startswith('default=') or line.startswith('default:'):
                default = line[8:].strip(' ')
                line = ''
            else:
                comma = line.index(',')
                if comma == -1:
                    raise Exception("Expected ',' in parameter.")
                line = line[comma+1:]
        index = index + 1
        attribute_lines = []
        while index < len(lines) and (len(lines[index].strip(' ')) == 0 or lines[index].startswith('        ')):
            attribute_lines.append(lines[index].lstrip(' '))
            index = index + 1
        description = '\n'.join(attribute_lines)
        update_attribute(schema, name, description, type, option, default)

for entry in json_root:
    name = entry['name']
    schema = entry['schema']
    if 'package' in schema:
        class_name = schema['package'] + '.' + name
        class_definition = pydoc.locate(class_name)
        if not class_definition:
            raise Exception('\'' + class_name + '\' not found.')
        docstring = class_definition.__doc__
        if not docstring:
            raise Exception('\'' + class_name + '\' missing __doc__.')
        headers = split_docstring(docstring)
        if '' in headers:
            update_description(schema, headers[''])
        if 'Parameters' in headers:
            update_attributes(schema, headers['Parameters'])

with io.open(json_file, 'w', newline='') as fout:
    json_data = json.dumps(json_root, sort_keys=True, indent=2)
    for line in json_data.splitlines():
        line = line.rstrip()
        if sys.version_info[0] < 3:
            line = unicode(line)
        fout.write(line)
        fout.write('\n')
