
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

    # with io.open(json_file, 'w', newline='') as fout:
    #     json_data = json.dumps(json_root, sort_keys=True, indent=2)
    #     for line in json_data.splitlines():
    #         line = line.rstrip()
    #         if sys.version_info[0] < 3:
    #             line = unicode(line)
    #         fout.write(line)
    #         fout.write('\n')

def rezip(zip_file_path):
    zip_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        zip_file.extractall(zip_dir)
    for folder, subfolders, files in os.walk(zip_dir):
        if folder.endswith('/data'):
            for file in files:
                path = os.path.join(folder, file)
                size = os.path.getsize(path)
                with open(path, 'w+b') as file_handle:
                    file_handle.write(bytearray(size))
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for folder, subfolders, files in os.walk(zip_dir):
            for file in files:
                path = os.path.join(folder, file)
                zip_file.write(path, os.path.relpath(path, zip_dir))
    shutil.rmtree(zip_dir)

def download_torchvision_model(zip_format, jit_format, traced_format, pretrained, name, input):
    folder = os.path.expandvars('${test}/data/pytorch')
    if not os.path.exists(folder):
        os.makedirs(folder)
    base = folder + '/' + name.split('.')[-1]
    model_type = pydoc.locate(name)
    model = model_type(pretrained=pretrained)
    import torch
    for param in model.parameters():
        param.data.fill_(0)
    if zip_format:
        file = base + '.zip.pth'
        torch.save(model, file, _use_new_zipfile_serialization=True)
        rezip(file)
    model.eval()
    if jit_format:
        file = base + '.pt'
        torch.jit.script(model).save(file)
        rezip(file)
    if traced_format:
        file = base + '_traced.pt'
        traced_model = torch.jit.trace(model, torch.rand(input))
        torch.jit.save(traced_model, file)
        rezip(file)

def zoo():
    if not os.environ.get('test'):
        os.environ['test'] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test'))
    download_torchvision_model(True , True , True , False, 'torchvision.models.alexnet', [ 1, 3, 299, 299 ])
    download_torchvision_model(True , True , True , False, 'torchvision.models.densenet161', [ 1, 3, 224, 224 ])
    download_torchvision_model(True , True , True , True,  'torchvision.models.inception_v3', [ 1, 3, 299, 299 ])
    download_torchvision_model(True , True , True , False, 'torchvision.models.mobilenet_v2', [ 1, 3, 224, 224 ])
    download_torchvision_model(False, True , True , False, 'torchvision.models.resnet101', [ 1, 3, 224, 224 ])
    download_torchvision_model(True , True , True , False, 'torchvision.models.shufflenet_v2_x1_0', [ 1, 3, 224, 224 ])
    download_torchvision_model(False, True , True , False, 'torchvision.models.squeezenet1_1', [ 1, 3, 224, 224 ])
    download_torchvision_model(True , True , True , False, 'torchvision.models.video.r3d_18', [ 1, 3, 4, 112, 112 ])

if __name__ == '__main__':
    command_table = { 'metadata': metadata, 'zoo': zoo }
    command = sys.argv[1]
    command_table[command]()
