#!/usr/bin/env python

''' Expermiental Python Server backend test '''

import os
import netron

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
third_party_dir = os.path.join(root_dir, 'third_party')

def _test_onnx():
    file = os.path.join(third_party_dir, 'test', 'onnx', 'candy.onnx')
    import onnx # pylint: disable=import-outside-toplevel
    model = onnx.load(file)
    netron.serve(None, model, browse=True, verbosity='quiet')

def _test_onnx_list():
    folder = os.path.join(third_party_dir, 'test', 'onnx')
    for item in os.listdir(folder):
        file = os.path.join(folder, item)
        if file.endswith('.onnx') and \
            item != 'super_resolution.onnx' and \
            item != 'arcface-resnet100.onnx':
            print(item)
            import onnx # pylint: disable=import-outside-toplevel
            model = onnx.load(file)
            address = netron.serve(file, model, verbosity='quiet')
            netron.stop(address)

def _test_torchscript():
    import torch # pylint: disable=import-outside-toplevel disable
    import torchvision # pylint: disable=import-outside-toplevel
    model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
    args = torch.zeros([1, 3, 224, 224]) # pylint: disable=no-member
    graph, _ = torch.jit._get_trace_graph(model, args) # pylint: disable=protected-access
    # graph = torch.onnx._optimize_trace(graph, torch.onnx.OperatorExportTypes.ONNX)
    # https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/ir/ir.h
    netron.serve('resnet34', graph, browse=True, verbosity='quiet')

# _test_onnx()
# _test_torchscript()
_test_onnx_list()
