
from __future__ import unicode_literals
from __future__ import print_function

import io
import json
import pydoc
import os
import re
import sys

def metadata():
    json_file = os.path.join(os.path.dirname(__file__), '../src/pytorch-metadata.json')
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

def download_torchvision_model(name, input):
    print(name)
    folder = os.path.expandvars('${test}/data/pytorch')
    if not os.path.exists(folder):
        os.makedirs(folder)
    base = folder + '/' + name.split('.')[-1]
    print(base)
    model = pydoc.locate(name)(pretrained=True)
    import torch
    torch.save(model, base + '.pkl.pth', _use_new_zipfile_serialization=False);
    torch.save(model, base + '.zip.pth', _use_new_zipfile_serialization=True);
    model.eval()
    torch.jit.script(model).save(base + '.pt')
    traced_model = torch.jit.trace(model, torch.rand(input))
    torch.jit.save(traced_model, base + '_traced.pt')

def zoo():
    if not os.environ.get('test'):
        os.environ['test'] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test'))
    download_torchvision_model('torchvision.models.alexnet', [ 1, 3, 299, 299 ])
    download_torchvision_model('torchvision.models.densenet161', [ 1, 3, 224, 224 ])
    download_torchvision_model('torchvision.models.inception_v3', [ 1, 3, 299, 299 ])
    download_torchvision_model('torchvision.models.mobilenet_v2', [ 1, 3, 224, 224 ])
    download_torchvision_model('torchvision.models.resnet18', [ 1, 3, 224, 224 ])
    download_torchvision_model('torchvision.models.resnet101', [ 1, 3, 224, 224 ])
    download_torchvision_model('torchvision.models.shufflenet_v2_x1_0', [ 1, 3, 224, 224 ])
    download_torchvision_model('torchvision.models.squeezenet1_1', [ 1, 3, 224, 224 ])
    download_torchvision_model('torchvision.models.vgg11_bn', [ 1, 3, 224, 224 ])
    download_torchvision_model('torchvision.models.vgg16', [ 1, 3, 224, 224 ])

if __name__ == '__main__':
    command_table = { 'metadata': metadata, 'zoo': zoo }
    command = sys.argv[1];
    command_table[command]()
