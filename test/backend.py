#!/usr/bin/env python

""" Expermiental Python Server backend test """

import logging
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.pycache_prefix = os.path.join(root_dir, "dist", "pycache", "backend")
netron = __import__("source")

third_party_dir = os.path.join(root_dir, "third_party")
test_data_dir = os.path.join(third_party_dir, "test")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

def _test_onnx():
    file = os.path.join(test_data_dir, "onnx", "candy.onnx")
    onnx = __import__("onnx")
    model = onnx.load(file)
    netron.serve(None, model)

def _test_onnx_iterate():
    logging.getLogger(netron.__name__).setLevel(logging.WARNING)
    folder = os.path.join(test_data_dir, "onnx")
    for item in os.listdir(folder):
        file = os.path.join(folder, item)
        skip = (
            "super_resolution.onnx",
            "arcface-resnet100.onnx",
            "aten_sum_dim_onnx_inlined.onnx",
            "phi3-mini-128k-instruct-cuda-fp16.onnx",
            "if_k1.onnx"
        )
        if file.endswith(".onnx") and item not in skip:
            logger.info(item)
            onnx = __import__("onnx")
            model = onnx.load(file)
            address = netron.serve(file, model)
            netron.stop(address)

def _test_torchscript(file):
    torch = __import__("torch")
    path = os.path.join(test_data_dir, "pytorch", file)
    model = torch.load(path, weights_only=False)
    torch._C._jit_pass_inline(model.graph)
    netron.serve(file, model)

def _test_torchscript_transformer():
    torch = __import__("torch")
    model = torch.nn.Transformer(nhead=16, num_encoder_layers=12)
    module = torch.jit.trace(model, (torch.rand(10, 32, 512), torch.rand(20, 32, 512)))
    # module = torch.jit.script(model)
    torch._C._jit_pass_inline(module.graph)
    netron.serve("transformer", module)

def _test_torchscript_resnet34():
    torch = __import__("torch")
    torchvision = __import__("torchvision")
    model = torchvision.models.resnet34()
    file = os.path.join(test_data_dir, "pytorch", "resnet34-333f7ec4.pth")
    state_dict = torch.load(file)
    model.load_state_dict(state_dict)
    trace = torch.jit.trace(model, torch.zeros([1, 3, 224, 224]), strict=True)
    torch._C._jit_pass_inline(trace.graph)
    netron.serve("resnet34", trace)

def _test_torchscript_quantized():
    torch = __import__("torch")
    __import__("torchvision")
    torch.backends.quantized.engine = "qnnpack"
    trace = torch.jit.load(os.path.join(test_data_dir, "pytorch", "d2go.pt"))
    torch._C._jit_pass_inline(trace.graph)
    netron.serve("d2go", trace)

# _test_onnx()
# _test_onnx_iterate()

# _test_torchscript('alexnet.pt')
_test_torchscript("gpt2.pt")
# _test_torchscript('inception_v3_traced.pt')
# _test_torchscript('netron_issue_920.pt') # scalar
# _test_torchscript('fasterrcnn_resnet50_fpn.pt') # tuple
# _test_torchscript('mobilenetv2-quant_full-nnapi.pt') # nnapi
# _test_torchscript_quantized()
# _test_torchscript_resnet34()
# _test_torchscript_transformer()
