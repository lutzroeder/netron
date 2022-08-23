''' Keras metadata script '''

import json
import os
import pydoc
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def _read(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

def _find_docstring(class_name):
    class_definition = pydoc.locate(class_name)
    if not class_definition:
        raise Exception('\'' + class_name + '\' not found.')
    if not class_definition.__doc__:
        raise Exception('\'' + class_name + '\' missing __doc__.')
    return class_definition.__doc__

def _parse_docstring(docstring):
    headers = []
    lines = docstring.splitlines()
    indentation = min(filter(lambda s: s > 0, map(lambda s: len(s) - len(s.lstrip()), lines)))
    lines = list((s[indentation:] if len(s) > len(s.lstrip()) else s) for s in lines)
    docstring = '\n'.join(lines)
    labels = [
        'Args', 'Arguments', 'Variables', 'Fields', 'Yields', 'Call arguments', 'Raises',
        'Examples', 'Example', 'Usage', 'Input shape', 'Output shape', 'Returns', 'References'
    ]
    tag_re = re.compile('(?<=\n)(' + '|'.join(labels) + '):\n', re.MULTILINE)
    parts = tag_re.split(docstring)
    headers.append(('', parts.pop(0)))
    while len(parts) > 0:
        headers.append((parts.pop(0), parts.pop(0)))
    return headers

def _parse_arguments(arguments):
    result = []
    item_re = re.compile(r'^   ? ?(\*?\*?\w[\w.]*?\s*):\s', re.MULTILINE)
    content = item_re.split(arguments)
    if content.pop(0) != '':
        raise Exception('')
    while len(content) > 0:
        result.append((content.pop(0), content.pop(0)))
    return result

def _convert_code_blocks(description):
    lines = description.splitlines()
    output = []
    while len(lines) > 0:
        line = lines.pop(0)
        if line.startswith('>>>') and len(lines) > 0 and \
            (lines[0].startswith('>>>') or lines[0].startswith('...')):
            output.append('```')
            output.append(line)
            while len(lines) > 0 and lines[0] != '':
                output.append(lines.pop(0))
            output.append('```')
        else:
            output.append(line)
    return '\n'.join(output)

def _remove_indentation(value):
    lines = value.splitlines()
    indentation = min(map(lambda s: len(s) - len(s.lstrip()), \
        filter(lambda s: len(s) > 0, lines)))
    lines = list((s[indentation:] if len(s) > 0 else s) for s in lines)
    return '\n'.join(lines).strip()

def _update_argument(schema, name, description):
    if not 'attributes' in schema:
        schema['attributes'] = []
    attribute = next((_ for _ in schema['attributes'] if _['name'] == name), None)
    if not attribute:
        attribute = {}
        attribute['name'] = name
        schema['attributes'].append(attribute)
    attribute['description'] = _remove_indentation(description)

def _update_input(schema, description):
    if not 'inputs' in schema:
        schema['inputs'] = [ { 'name': 'input' } ]
    parameter = next((_ for _ in schema['inputs'] \
        if (_['name'] == 'input' or _['name'] == 'inputs')), None)
    if parameter:
        parameter['description'] = _remove_indentation(description)
    else:
        raise Exception('')

def _update_output(schema, description):
    if not 'outputs' in schema:
        schema['outputs'] = [ { 'name': 'output' } ]
    parameter = next((param for param in schema['outputs'] if param['name'] == 'output'), None)
    if parameter:
        parameter['description'] = _remove_indentation(description)
    else:
        raise Exception('')

def _update_examples(schema, value):
    if 'examples' in schema:
        del schema['examples']
    value = _convert_code_blocks(value)
    lines = value.splitlines()
    code = []
    summary = []
    while len(lines) > 0:
        line = lines.pop(0)
        if len(line) > 0:
            if line.startswith('```'):
                while len(lines) > 0:
                    line = lines.pop(0)
                    if line == '```':
                        break
                    code.append(line)
            else:
                summary.append(line)
        if len(code) > 0:
            example = {}
            if len(summary) > 0:
                example['summary'] = '\n'.join(summary)
            example['code'] = '\n'.join(code)
            if not 'examples' in schema:
                schema['examples'] = []
            schema['examples'].append(example)
            code = []
            summary = []

def _update_references(schema, value):
    if 'references' in schema:
        del schema['references']
    references = []
    reference = ''
    lines = value.splitlines()
    for line in lines:
        if line.lstrip().startswith('- '):
            if len(reference) > 0:
                references.append(reference)
            reference = line.lstrip().lstrip('- ')
        else:
            if line.startswith('  '):
                line = line[2:]
            reference = ' '.join([ reference, line.strip() ])
    if len(reference) > 0:
        references.append(reference)
    for reference in references:
        if not 'references' in schema:
            schema['references'] = []
        schema['references'].append({ 'description': reference })

def _update_headers(schema, docstring):
    headers = _parse_docstring(docstring)
    for header in headers:
        key = header[0]
        value = header[1]
        if key == '':
            description = _convert_code_blocks(value)
            schema['description'] = _remove_indentation(description)
        elif key in ('Args', 'Arguments'):
            arguments = _parse_arguments(value)
            for argument in arguments:
                _update_argument(schema, argument[0], argument[1])
        elif key == 'Input shape':
            _update_input(schema, value)
        elif key == 'Output shape':
            _update_output(schema, value)
        elif key in ('Example', 'Examples', 'Usage'):
            _update_examples(schema, value)
        elif key == 'References':
            _update_references(schema, value)
        elif key in ('Call arguments', 'Returns', 'Variables', 'Raises'):
            pass
        else:
            raise Exception('')


def _metadata():

    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    json_path = os.path.join(root, 'source', 'keras-metadata.json')
    json_root = json.loads(_read(json_path))

    for schema in json_root:
        if 'module' in schema:
            class_name = schema['module'] + '.' + schema['name']
            docstring = _find_docstring(class_name)
            _update_headers(schema, docstring)

    with open(json_path, 'w', encoding='utf-8') as file:
        content = json.dumps(json_root, sort_keys=False, indent=2)
        for line in content.splitlines():
            file.write(line.rstrip() + '\n')

def main(): # pylint: disable=missing-function-docstring
    _metadata()

if __name__ == '__main__':
    main()
