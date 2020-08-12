
from __future__ import unicode_literals
from __future__ import print_function

import io
import json
import os
import pydoc
import re
import sys

json_file = os.path.join(os.path.dirname(__file__), '../source/sklearn-metadata.json')
json_data = open(json_file).read()
json_root = json.loads(json_data)

def split_docstring(docstring):
    headers = {}
    current_header = ''
    current_lines = []
    lines = docstring.split('\n')
    index = 0
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

def update_attribute(schema, name, description, attribute_type, option, default_value):
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
    if attribute_type:
        attribute['type'] = attribute_type
    if option:
        attribute['option'] = option
    if default_value:
        if attribute_type == 'float32':
            if default_value == 'None':
                attribute['default'] = None
            elif default_value != "'auto'":
                attribute['default'] = float(default_value)
            else:
                attribute['default'] = default_value.strip("'").strip('"')
        elif attribute_type == 'int32':
            if default_value == 'None':
                attribute['default'] = None
            elif default_value == "'auto'" or default_value == '"auto"':
                attribute['default'] = default_value.strip("'").strip('"')
            else:
                attribute['default'] = int(default_value)
        elif attribute_type == 'string':
            attribute['default'] = default_value.strip("'").strip('"')
        elif attribute_type == 'boolean':
            if default_value == 'True':
                attribute['default'] = True
            elif default_value == 'False':
                attribute['default'] = False
            elif default_value == "'auto'":
                attribute['default'] = default_value.strip("'").strip('"')
            else:
                raise Exception("Unknown boolean default value '" + str(default_value) + "'.")
        else:
            if attribute_type:
                raise Exception("Unknown default type '" + attribute_type + "'.")
            else:
                if default_value == 'None':
                    attribute['default'] = None
                else:
                    attribute['default'] = default_value.strip("'")

def update_attributes(schema, lines):
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.endswith('.'):
            line = line[0:-1]
        colon = line.find(':')
        if colon == -1:
            raise Exception("Expected ':' in parameter.")
        name = line[0:colon].strip(' ')
        line = line[colon + 1:].strip(' ')
        attribute_type = None
        type_map = { 'float': 'float32', 'boolean': 'boolean', 'bool': 'boolean', 'string': 'string', 'int': 'int32', 'integer': 'int32' }
        skip_map = {
            "'sigmoid' or 'isotonic'",
            'instance BaseEstimator',
            'callable or None (default)',
            'str or callable',
            "string {'english'}, list, or None (default)",
            'tuple (min_n, max_n)',
            "string, {'word', 'char', 'char_wb'} or callable",
            "{'word', 'char'} or callable",
            "string, {'word', 'char'} or callable",
            'int, float, None or string',
            "int, float, None or str",
            "int or None, optional (default=None)",
            "'l1', 'l2' or None, optional",
            "{'strict', 'ignore', 'replace'} (default='strict')",
            "{'ascii', 'unicode', None} (default=None)",
            "string {'english'}, list, or None (default=None)",
            "tuple (min_n, max_n) (default=(1, 1))",
            "float in range [0.0, 1.0] or int (default=1.0)",
            "float in range [0.0, 1.0] or int (default=1)",
            "'l1', 'l2' or None, optional (default='l2')",
            "{'scale', 'auto'} or float, optional (default='scale')",
            "str {'auto', 'full', 'arpack', 'randomized'}",
            "str {'filename', 'file', 'content'}",
            "str, {'word', 'char', 'char_wb'} or callable",
            "str {'english'}, list, or None (default=None)",
            "{'scale', 'auto'} or float, optional (default='scale')",
            "{'word', 'char', 'char_wb'} or callable, default='word'",
            "{'scale', 'auto'} or float, default='scale'",
            "{'uniform', 'distance'} or callable, default='uniform'",
            "int, RandomState instance or None (default)",
            "list of (string, transformer) tuples",
            "list of tuples",
            "{'drop', 'passthrough'} or estimator, default='drop'",
            "'auto' or a list of array-like, default='auto'",
            "{'first', 'if_binary'} or a array-like of shape (n_features,),             default=None",
            "callable",
            "int or \"all\", optional, default=10",
            "number, string, np.nan (default) or None",
            "estimator object",
            "dict or list of dictionaries",
            "int, or str, default=n_jobs",
            "'raise' or numeric, default=np.nan",
            "'auto' or float, default=None",
            "float, default=np.finfo(float).eps",
            "int, float, str, np.nan or None, default=np.nan"
        }
        if line == 'str':
            line = 'string'
        if line in skip_map:
            line = ''
        elif line.startswith('{'):
            if line.endswith('}'):
                line = ''
            else:
                end = line.find('},')
                if end == -1:
                    raise Exception("Expected '}' in parameter.")
                # attribute_type = line[0:end + 1]
                line = line[end + 2:].strip(' ')
        elif line.startswith("'"):
            while line.startswith("'"):
                end = line.find("',")
                if end == -1:
                    raise Exception("Expected \' in parameter.")
                line = line[end + 2:].strip(' ')
        elif line in type_map:
            attribute_type = line
            line = ''
        elif line.startswith('int, RandomState instance or None,'):
            line = line[len('int, RandomState instance or None,'):]
        elif line.find('|') != -1:
            line = ''
        else:
            space = line.find(' {')
            if space != -1 and line[0:space] in type_map and line[space:].find('}') != -1:
                attribute_type = line[0:space]
                end = line[space:].find('}')
                line = line[space+end+1:]
            else:
                comma = line.find(',')
                if comma == -1:
                    comma = line.find(' (')
                    if comma == -1:
                        raise Exception("Expected ',' in parameter.")
                attribute_type = line[0:comma]
                line = line[comma + 1:].strip(' ')
        if attribute_type in type_map:
            attribute_type = type_map[attribute_type]
        else:
            attribute_type = None
        # elif type == "{dict, 'balanced'}":
        #    v = 'map'
        # else:
        #    raise Exception("Unknown attribute type '" + attribute_type + "'.")
        option = None
        default = None
        while len(line.strip(' ')) > 0:
            line = line.strip(' ')
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
            elif line.startswith('default =') or line.startswith('default :'):
                default = line[9:].strip(' ')
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
        update_attribute(schema, name, description, attribute_type, option, default)

for entry in json_root:
    name = entry['name']
    entry['schema'] = entry['schema'] if 'schema' in entry else {}
    schema = entry['schema']
    skip_modules = [
        'lightgbm.',
        'sklearn.svm.classes',
        'sklearn.ensemble.forest.',
        'sklearn.ensemble.weight_boosting.',
        'sklearn.neural_network.multilayer_perceptron.',
        'sklearn.tree.tree.'
    ]
    if not any(name.startswith(module) for module in skip_modules):
        class_definition = pydoc.locate(name)
        if not class_definition:
            raise Exception('\'' + name + '\' not found.')
        docstring = class_definition.__doc__
        if not docstring:
            raise Exception('\'' + name + '\' missing __doc__.')
        headers = split_docstring(docstring)
        if '' in headers:
            update_description(schema, headers[''])
        if 'Parameters' in headers:
            update_attributes(schema, headers['Parameters'])

with io.open(json_file, 'w', newline='') as fout:
    json_data = json.dumps(json_root, sort_keys=True, indent=2)
    for line in json_data.splitlines():
        fout.write(line.rstrip())
        fout.write('\n')
