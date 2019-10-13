
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

def download_pytorch_model(type, file):
    file = os.path.expandvars(file)
    if not os.path.exists(file):
        folder = os.path.dirname(file);
        if not os.path.exists(folder):
            os.makedirs(folder)
        import torch
        model = pydoc.locate(type)(pretrained=True)
        torch.save(model, file);

def download_torchscript_model(type, file):
    file = os.path.expandvars(file)
    if not os.path.exists(file):
        folder = os.path.dirname(file);
        if not os.path.exists(folder):
            os.makedirs(folder)
        import torch
        model = pydoc.locate(type)(pretrained=True)
        model.eval()
        torch.jit.script(model).save(file)

def download_torchscript_traced_model(type, file, input):
    file = os.path.expandvars(file)
    if not os.path.exists(file):
        folder = os.path.dirname(file);
        if not os.path.exists(folder):
            os.makedirs(folder)
        import torch
        model = pydoc.locate(type)(pretrained=True)
        model.eval()
        traced_model = torch.jit.trace(model, torch.rand(input))
        torch.jit.save(traced_model, file)

def zoo():
    if not os.environ.get('test'):
        os.environ['test'] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test'))
    download_pytorch_model('torchvision.models.alexnet', '${test}/data/pytorch/alexnet.pth')
    download_pytorch_model('torchvision.models.densenet121', '${test}/data/pytorch/densenet121.pth')
    download_pytorch_model('torchvision.models.densenet161', '${test}/data/pytorch/densenet161.pth')
    download_pytorch_model('torchvision.models.inception_v3', '${test}/data/pytorch/inception_v3.pth')
    download_pytorch_model('torchvision.models.mobilenet_v2', '${test}/data/pytorch/mobilenet_v2.pth')
    download_pytorch_model('torchvision.models.resnet18', '${test}/data/pytorch/resnet18.pth')
    download_pytorch_model('torchvision.models.resnet50', '${test}/data/pytorch/resnet50.pth')
    download_pytorch_model('torchvision.models.resnet101', '${test}/data/pytorch/resnet101.pth')
    download_pytorch_model('torchvision.models.squeezenet1_0', '${test}/data/pytorch/squeezenet1_0.pth')
    download_pytorch_model('torchvision.models.vgg11_bn', '${test}/data/pytorch/vgg11_bn.pth')
    download_pytorch_model('torchvision.models.vgg16', '${test}/data/pytorch/vgg16.pth')
    download_torchscript_model('torchvision.models.alexnet', '${test}/data/torchscript/alexnet.pt')
    download_torchscript_traced_model('torchvision.models.alexnet', '${test}/data/torchscript/alexnet_traced.pt', [ 1, 3, 299, 299 ])
    download_torchscript_traced_model('torchvision.models.densenet121', '${test}/data/torchscript/densenet121_traced.pt', [ 1, 3, 224, 224 ])
    download_torchscript_traced_model('torchvision.models.inception_v3', '${test}/data/torchscript/inception_v3_traced.pt', [ 1, 3, 299, 299 ])
    download_torchscript_traced_model('torchvision.models.mobilenet_v2', '${test}/data/torchscript/mobilenet_v2_traced.pt', [ 1, 3, 224, 224 ])
    download_torchscript_traced_model('torchvision.models.resnet18', '${test}/data/torchscript/resnet18_traced.pt', [ 1, 3, 224, 224 ])
    download_torchscript_traced_model('torchvision.models.resnet50', '${test}/data/torchscript/resnet50_traced.pt', [ 1, 3, 224, 224 ])
    download_torchscript_traced_model('torchvision.models.squeezenet1_1', '${test}/data/torchscript/squeezenet1_1_traced.pt', [ 1, 3, 224, 224 ])
    download_torchscript_traced_model('torchvision.models.vgg16', '${test}/data/torchscript/vgg16_traced.pt', [ 1, 3, 224, 224 ])

if __name__ == '__main__':
    command_table = { 'metadata': metadata, 'zoo': zoo }
    command = sys.argv[1];
    command_table[command]()
