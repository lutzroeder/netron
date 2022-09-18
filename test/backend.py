#!/usr/bin/env python

''' Expermiental Python Server backend test '''

import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.pycache_prefix = os.path.join(root_dir, 'dist', '__pycache__')
netron = __import__('source')

test_data_dir = os.path.join(root_dir, 'third_party', 'test')

def _test_onnx():
    file = os.path.join(test_data_dir, 'onnx', 'candy.onnx')
    onnx = __import__('onnx')
    model = onnx.load(file)
    netron.serve(None, model, browse=True, verbosity='quiet')

def _test_onnx_list():
    folder = os.path.join(test_data_dir, 'onnx')
    for item in os.listdir(folder):
        file = os.path.join(folder, item)
        if file.endswith('.onnx') and \
            item != 'super_resolution.onnx' and \
            item != 'arcface-resnet100.onnx':
            print(item)
            onnx = __import__('onnx')
            model = onnx.load(file)
            address = netron.serve(file, model, verbosity='quiet')
            netron.stop(address)

def _test_torchscript():
    torch = __import__('torch')
    torchvision = __import__('torchvision')
    model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
    args = torch.zeros([1, 3, 224, 224])
    graph, _ = torch.jit._get_trace_graph(model, args) # pylint: disable=protected-access
    # graph = torch.onnx._optimize_trace(graph, torch.onnx.OperatorExportTypes.ONNX)
    # https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/ir/ir.h
    netron.serve('resnet34', graph, browse=True, verbosity='quiet')

# _test_onnx()
# _test_torchscript()
_test_onnx_list()
