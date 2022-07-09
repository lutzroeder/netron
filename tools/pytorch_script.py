''' PyTorch metadata script '''

import json
import pydoc
import os
import sys

def metadata():
    ''' Update PyTorch metadata file '''
    json_file = os.path.join(os.path.dirname(__file__), '../source/pytorch-metadata.json')
    with open(json_file, 'r', encoding='utf-8') as file:
        json_root = json.loads(file.read())

    schema_map = {}

    for schema in json_root:
        name = schema['name']
        schema_map[name] = schema

    for schema in json_root:
        name = schema['name']
        if 'module' in schema:
            class_name = schema['module'] + '.' + name
            # print(class_name)
            class_definition = pydoc.locate(class_name)
            if not class_definition:
                raise Exception('\'' + class_name + '\' not found.')
            if not class_definition.__doc__:
                raise Exception('\'' + class_name + '\' missing __doc__.')
            # print(class_definition.__doc__)

def main(): # pylint: disable=missing-function-docstring
    command_table = { 'metadata': metadata }
    command = sys.argv[1]
    command_table[command]()

if __name__ == '__main__':
    main()
