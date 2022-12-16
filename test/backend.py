#!/usr/bin/env python

''' Expermiental Python Server backend test '''

import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.pycache_prefix = os.path.join(root_dir, 'dist', 'pycache', 'test', 'backend')
netron = __import__('source')

third_party_dir = os.path.join(root_dir, 'third_party')
test_data_dir = os.path.join(third_party_dir, 'test')

def _test_onnx():
    file = os.path.join(test_data_dir, 'onnx', 'candy.onnx')
    onnx = __import__('onnx')
    model = onnx.load(file)
    netron.serve(None, model)

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
    # model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
    # model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
    model = torchvision.models.resnet34()
    state_dict = torch.load(os.path.join(test_data_dir, 'pytorch', 'resnet34-333f7ec4.pth'))
    model.load_state_dict(state_dict)
    args = torch.zeros([1, 3, 224, 224])
    trace = torch.jit.trace(model, args, strict=True)
    # graph, _ = torch.jit._get_trace_graph(model, args) # pylint: disable=protected-access
    # torch.onnx._optimize_trace(graph, torch.onnx.OperatorExportTypes.ONNX)
    # trace = torch.load(os.path.join(test_data_dir, 'pytorch', 'fasterrcnn_resnet50_fpn.pt'))
    # torch.backends.quantized.engine = 'qnnpack'
    # trace = torch.load(os.path.join(test_data_dir, 'pytorch', 'd2go.pt'))
    # trace = torch.load(os.path.join(test_data_dir, 'pytorch', 'mobilenetv2-quant_full-nnapi.pt'))
    # trace = torch.load(os.path.join(test_data_dir, 'pytorch', 'inception_v3_traced.pt'))
    # trace = torch.load(os.path.join(test_data_dir, 'pytorch', 'netron_issue_920.pt'))
    # trace = torch.load(os.path.join(test_data_dir, 'pytorch', 'bert-base-uncased.pt'))
    # trace = torch.load(os.path.join(test_data_dir, 'pytorch', 'UNet.pt'))
    torch._C._jit_pass_inline(trace.graph)
    netron.serve('resnet34', trace)

# _test_onnx()
_test_torchscript()
# _test_onnx_list()
