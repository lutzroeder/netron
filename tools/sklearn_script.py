''' scikit-learn metadata script '''

import json
import os
import pydoc
import re
import sys

def _split_docstring(value):
    headers = {}
    current_header = ''
    current_lines = []
    lines = value.split('\n')
    index = 0
    while index < len(lines):
        if index + 1 < len(lines) and len(lines[index + 1].strip(' ')) > 0 and \
            len(lines[index + 1].strip(' ').strip('-')) == 0:
            headers[current_header] = current_lines
            current_header = lines[index].strip(' ')
            current_lines = []
            index = index + 1
        else:
            current_lines.append(lines[index])
        index = index + 1
    headers[current_header] = current_lines
    return headers

def _update_description(schema, lines):
    if len(''.join(lines).strip(' ')) > 0:
        for i, value in enumerate(lines):
            lines[i] = value.lstrip(' ')
        schema['description'] = '\n'.join(lines)

def _attribute_value(attribute_type, attribute_value):
    if attribute_value in ('None', 'np.finfo(float).eps'):
        return None
    if attribute_type in ('float32', 'int32', 'boolean', 'string'):
        if attribute_value in ("'auto'", '"auto"') or attribute_type == 'string':
            return attribute_value.strip("'").strip('"')
    if attribute_type == 'float32':
        return float(attribute_value)
    if attribute_type == 'int32':
        return int(attribute_value)
    if attribute_type == 'boolean':
        if attribute_value in ('True', 'False'):
            return attribute_value == 'True'
        raise ValueError("Unknown boolean default value '" + str(attribute_value) + "'.")
    if attribute_type:
        raise ValueError("Unknown default type '" + attribute_type + "'.")
    return attribute_value.strip("'")

def _find_attribute(schema, name):
    schema.setdefault('attributes', [])
    attribute = next((_ for _ in schema['attributes'] if _['name'] == name), None)
    if not attribute:
        attribute = { 'name': name }
        schema['attributes'].append(attribute)
    return attribute

def _update_attributes(schema, lines):
    doc_indent = '    ' if sys.version_info[:2] >= (3, 13) else '        '
    while len(lines) > 0:
        line = lines.pop(0)
        match = re.match(r'\s*(\w*)\s*:\s*(.*)\s*', line)
        if not match:
            raise SyntaxError("Expected ':' in parameter.")
        name = match.group(1)
        line = match.group(2)
        attribute = _find_attribute(schema, name)
        match = re.match(r'(.*),\s*default=(.*)\s*', line)
        default_value = None
        if match:
            line = match.group(1)
            default_value = match.group(2)
        attribute_types = {
            'float': 'float32',
            'boolean': 'boolean',
            'bool': 'boolean',
            'str': 'string',
            'string': 'string',
            'int': 'int32',
            'integer': 'int32'
        }
        attribute_type = attribute_types.get(line, None)
        if default_value:
            attribute['default'] = _attribute_value(attribute_type, default_value)
        description = []
        while len(lines) > 0 and (len(lines[0].strip(' ')) == 0 or lines[0].startswith(doc_indent)):
            line = lines.pop(0).lstrip(' ')
            description.append(line)
        attribute['description'] = '\n'.join(description)

def _metadata():
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    json_file = os.path.join(root_dir, 'source', 'sklearn-metadata.json')
    with open(json_file, 'r', encoding='utf-8') as file:
        json_root = json.loads(file.read())

    for schema in json_root:
        name = schema['name']
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
                raise KeyError('\'' + name + '\' not found.')
            docstring = class_definition.__doc__
            if not docstring:
                raise Exception('\'' + name + '\' missing __doc__.') # pylint: disable=broad-exception-raised
            headers = _split_docstring(docstring)
            if '' in headers:
                _update_description(schema, headers[''])
            if 'Parameters' in headers:
                _update_attributes(schema, headers['Parameters'])

    with open(json_file, 'w', encoding='utf-8') as file:
        file.write(json.dumps(json_root, sort_keys=False, indent=2))

def main(): # pylint: disable=missing-function-docstring
    _metadata()

if __name__ == '__main__':
    main()
