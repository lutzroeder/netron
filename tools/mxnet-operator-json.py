#!/usr/bin/env python

from __future__ import unicode_literals
from __future__ import print_function

import io
import json
import pydoc
import re
import sys

json_file = '../src/mxnet-operator.json'
json_data = open(json_file).read()
json_root = json.loads(json_data)

for entry in json_root:
    name = entry['name']
    schema = entry['schema']
    class_name = 'mxnet.symbol.' + name
    class_definition = pydoc.locate(class_name)
    if not class_definition:
        print('NOT FOUND: ' + class_name)
        # raise Exception('\'' + class_name + '\' not found.')
    else:
        docstring = class_definition.__doc__
        if docstring:
            schema['description'] = docstring
    # if not docstring:
        # print('NO DOCSTRING: ' + class_name)
        # raise Exception('\'' + class_name + '\' missing __doc__.')
    # print(docstring)
 
with io.open(json_file, 'w', newline='') as fout:
    json_data = json.dumps(json_root, sort_keys=True, indent=2)
    for line in json_data.splitlines():
        line = line.rstrip()
        if sys.version_info[0] < 3:
            line = unicode(line)
        fout.write(line)
        fout.write('\n')

