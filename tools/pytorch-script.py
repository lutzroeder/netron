
from __future__ import unicode_literals
from __future__ import print_function

import io
import json
import pydoc
import os
import re
import sys

def metadata():
    json_file = '../src/pytorch-metadata.json'
    json_data = open(json_file).read()
    json_root = json.loads(json_data)

    schema_map = {}

    for entry in json_root:
        name = entry['name']
        schema = entry['schema']
        schema_map[name] = schema

    for entry in json_root:
        name = entry['name']
        schema = entry['schema']
        if 'package' in schema:
            class_name = schema['package'] + '.' + name
            # print(class_name)
            class_definition = pydoc.locate(class_name)
            if not class_definition:
                raise Exception('\'' + class_name + '\' not found.')
            docstring = class_definition.__doc__
            if not docstring:
                raise Exception('\'' + class_name + '\' missing __doc__.')
            # print(docstring)

    with io.open(json_file, 'w', newline='') as fout:
        json_data = json.dumps(json_root, sort_keys=True, indent=2)
        for line in json_data.splitlines():
            line = line.rstrip()
            if sys.version_info[0] < 3:
                line = unicode(line)
            fout.write(line)
            fout.write('\n')

def zoo():
    import torch
    type = sys.argv[2];
    file = sys.argv[3];
    directory = os.path.dirname(file);
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(type)
    model = pydoc.locate(type)(pretrained=True)
    torch.save(model, file);

if __name__ == '__main__':
    command_table = { 'metadata': metadata, 'zoo': zoo }
    command = sys.argv[1];
    command_table[command]()
