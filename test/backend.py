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

def _test_onnx_iterate():
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

def _test_torchscript_transformer():
    torch = __import__('torch')
    model = torch.nn.Transformer(nhead=16, num_encoder_layers=12)
    module = torch.jit.trace(model, (torch.rand(10, 32, 512), torch.rand(20, 32, 512)))
    # module = torch.jit.script(model)
    torch._C._jit_pass_inline(module.graph) # pylint: disable=protected-access
    netron.serve('transformer', module)

def _test_torchscript_resnet34():
    torch = __import__('torch')
    torchvision = __import__('torchvision')
    model = torchvision.models.resnet34()
    # model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
    # model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
    state_dict = torch.load(os.path.join(test_data_dir, 'pytorch', 'resnet34-333f7ec4.pth'))
    model.load_state_dict(state_dict)
    trace = torch.jit.trace(model, torch.zeros([1, 3, 224, 224]), strict=True)
    torch._C._jit_pass_inline(trace.graph) # pylint: disable=protected-access
    netron.serve('resnet34', trace)

def _test_torchscript_quantized():
    torch = __import__('torch')
    __import__('torchvision')
    torch.backends.quantized.engine = 'qnnpack'
    trace = torch.jit.load(os.path.join(test_data_dir, 'pytorch', 'd2go.pt'))
    torch._C._jit_pass_inline(trace.graph) # pylint: disable=protected-access
    netron.serve('d2go', trace)

def _test_torchscript_inception_v3():
    torch = __import__('torch')
    trace = torch.jit.load(os.path.join(test_data_dir, 'pytorch', 'inception_v3_traced.pt'))
    torch._C._jit_pass_inline(trace.graph) # pylint: disable=protected-access
    netron.serve('inception_v3', trace)

def _test_torchscript_scalar():
    torch = __import__('torch')
    trace = torch.load(os.path.join(test_data_dir, 'pytorch', 'netron_issue_920.pt'))
    # trace = torch.load(os.path.join(test_data_dir, 'pytorch', 'UNet.pt'))
    torch._C._jit_pass_inline(trace.graph) # pylint: disable=protected-access
    netron.serve('inception_v3', trace)

def _test_torchscript_tuple():
    torch = __import__('torch')
    __import__('torchvision')
    trace = torch.load(os.path.join(test_data_dir, 'pytorch', 'fasterrcnn_resnet50_fpn.pt'))
    torch._C._jit_pass_inline(trace.graph) # pylint: disable=protected-access
    netron.serve('inception_v3', trace)

def _test_torchscript_nnapi():
    torch = __import__('torch')
    trace = torch.load(os.path.join(test_data_dir, 'pytorch', 'mobilenetv2-quant_full-nnapi.pt'))
    torch._C._jit_pass_inline(trace.graph) # pylint: disable=protected-access
    netron.serve('inception_v3', trace)

# _test_onnx()
# _test_onnx_iterate()

# _test_torchscript()
# _test_torchscript_quantized()
# _test_torchscript_resnet34()
# _test_torchscript_inception_v3()
# _test_torchscript_scalar()
# _test_torchscript_tuple()
# _test_torchscript_nnapi()
_test_torchscript_transformer()
