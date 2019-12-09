
from __future__ import unicode_literals
from __future__ import print_function

import io
import json
import os
import re
import sys

def metadata():
    json_file = os.path.join(os.path.dirname(__file__), '../src/mlnet-metadata.json')
    json_data = open(json_file).read()
    json_root = json.loads(json_data)
    manifest_file = os.path.join(os.path.dirname(__file__), '../third_party/mlnet/test/BaselineOutput/Common/EntryPoints/core_manifest.json')
    manifest_data = open(manifest_file).read()
    manifest_root = json.loads(manifest_data)
    schema_map = {}
    # for manifest in manifest_root['EntryPoints']:
    #     print(manifest['Name'])

if __name__ == '__main__':
    command_table = { 'metadata': metadata }
    command = sys.argv[1];
    command_table[command]()
