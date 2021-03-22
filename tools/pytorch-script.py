
from __future__ import unicode_literals
from __future__ import print_function

import io
import json
import pydoc
import os
import re
import shutil
import sys
import tempfile
import zipfile

def metadata():
    json_file = os.path.join(os.path.dirname(__file__), '../source/pytorch-metadata.json')
    json_data = open(json_file).read()
    json_root = json.loads(json_data)

    schema_map = {}

    for schema in json_root:
        name = schema['name']
        schema_map[name] = schema

    for schema in json_root:
        name = schema['name']
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

    # with io.open(json_file, 'w', newline='') as fout:
    #     json_data = json.dumps(json_root, sort_keys=True, indent=2)
    #     for line in json_data.splitlines():
    #         line = line.rstrip()
    #         if sys.version_info[0] < 3:
    #             line = unicode(line)
    #         fout.write(line)
    #         fout.write('\n')

if __name__ == '__main__':
    command_table = { 'metadata': metadata }
    command = sys.argv[1]
    command_table[command]()
