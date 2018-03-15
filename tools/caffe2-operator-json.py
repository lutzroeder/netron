#!/usr/bin/env python

from __future__ import unicode_literals
from __future__ import print_function

import io
import json
import pydoc
import re
import sys
import caffe2.python.core
import caffe2.python.workspace

json_file = '../src/caffe2-operator.json'
json_data = open(json_file).read()
json_root = json.loads(json_data)

def filePriority(x):
    if x == "caffe2/caffe2/operators":
        return 0
    if 'contrib' in x.split('/'):
        return 2
    if 'experiments' in x.split('/'):
        return 3
    return 1

for name in caffe2.python.core._GetRegisteredOperators():
    print(name)
    schema = caffe2.python.workspace.C.OpSchema.get(name)
    if schema:
        print('  ' + schema.file)
        print('  ' + schema.doc)
        # schema.file
        priority = filePriority(os.path.dirname(schema.file))
        # schema.args[0]
        # schema.args[0].name
        # schema.args[0].description
        # schema.input_desc[0][0] ==> name
        # schema.input_desc[0][1] ==> description
        # schema.output_desc

# with io.open(json_file, 'w', newline='') as fout:
#     json_data = json.dumps(json_root, sort_keys=True, indent=2)
#     for line in json_data.splitlines():
#         line = line.rstrip()
#         if sys.version_info[0] < 3:
#             line = unicode(line)
#         fout.write(line)
#         fout.write('\n')
