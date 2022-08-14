#!/usr/bin/env python

''' Expermiental Python Server backend test '''

import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
source_dir = os.path.join(root_dir, 'source')
third_party_dir = os.path.join(root_dir, 'third_party')
package_dir = os.path.join(root_dir, 'dist', 'backend')
if not os.path.exists(package_dir):
    os.makedirs(package_dir)
    os.symlink(source_dir, package_dir + '/netron')
sys.path.append(package_dir)

import netron # pylint: disable=wrong-import-position disable=import-error

def _test_onnx():
    file = os.path.join(third_party_dir, 'test', 'onnx', 'candy.onnx')
    import onnx # pylint: disable=import-outside-toplevel
    model = onnx.load(file)
    netron.serve('x.onnx', model, browse=True)

def _test_onnx_list():
    folder = os.path.join(third_party_dir, 'test', 'onnx')
    for item in os.listdir(folder):
        file = os.path.join(folder, item)
        if file.endswith('.onnx') and item != 'super_resolution.onnx':
            print(item)
            import onnx # pylint: disable=import-outside-toplevel
            model = onnx.load(file)
            address = netron.serve('x.onnx', model)
            netron.stop(address)
            print()

def _test_torchscript():
    import torch # pylint: disable=import-outside-toplevel disable
    import torchvision # pylint: disable=import-outside-toplevel
    model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
    args = torch.zeros([1, 3, 224, 224]) # pylint: disable=no-member
    graph = torch.jit._get_trace_graph(model, args) # pylint: disable=protected-access
    # graph = torch.onnx._optimize_trace(graph, torch.onnx.OperatorExportTypes.ONNX)
    # https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/ir/ir.h
    netron.serve('x.pt', graph, browse=True)

# _test_onnx()
# _test_torchscript()
_test_onnx_list()
