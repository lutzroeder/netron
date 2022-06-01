
def serialize(model):
    print('Experimental')
    import onnx.shape_inference
    model = onnx.shape_inference.infer_shapes(model)
    import onnx.onnx_pb
    json_model = {}
    json_model['signature'] = 'netron:onnx'
    json_model['format'] = 'ONNX' + (' v' + str(model.ir_version) if model.ir_version else '')
    if model.producer_name and len(model.producer_name) > 0:
        json_model['producer'] = model.producer_name + (' v' + model.producer_version if model.producer_version else '')
    if model.model_version and model.model_version != 0:
        json_model['version'] = str(model.model_version)
    if model.doc_string and len(model.doc_string):
        json_model['description'] = str(model.doc_string)
    json_model['graphs'] = []
    graph = model.graph
    json_graph = {}
    json_graph['nodes'] = []
    json_model['graphs'].append(json_graph)
    for node in graph.node:
        json_node = {}
        json_node_type = {}
        json_node_type['name'] = node.op_type
        if category(node.op_type):
            json_node_type['category'] = category(node.op_type)
        json_node['type'] = json_node_type
        if node.name:
            json_node['name'] = node.name
        json_node['inputs'] = []
        for input in node.input:
            json_node['inputs'].append({
                    'name': '',
                    'arguments': [ { 'name': input } ]
                })
        json_node['outputs'] = []
        for output in node.output:
            json_node['outputs'].append({
                    'name': '',
                    'arguments': [ { 'name': output } ]
                })
        json_node['attributes'] = []
        for attribute in node.attribute:
            json_attribute = {}
            json_attribute['name'] = attribute.name
            if attribute.type == onnx.onnx_pb.AttributeProto.FLOAT:
                json_attribute['type'] = 'float32'
                json_attribute['value'] = attribute.f
            elif attribute.type == onnx.onnx_pb.AttributeProto.INT:
                json_attribute['type'] = 'int64'
                json_attribute['value'] = attribute.i
            elif attribute.type == onnx.onnx_pb.AttributeProto.STRING:
                json_attribute['type'] = 'string'
                json_attribute['value'] = attribute.s.decode('utf-8')
            elif attribute.type == onnx.onnx_pb.AttributeProto.TENSOR:
                raise Exception('Unsupported tensor attribute type')
            elif attribute.type == onnx.onnx_pb.AttributeProto.GRAPH:
                raise Exception('Unsupported graph attribute type')
            elif attribute.type == onnx.onnx_pb.AttributeProto.FLOATS:
                json_attribute['type'] = 'float32[]'
                json_attribute['value'] = [ item for item in attribute.floats ]
            elif attribute.type == onnx.onnx_pb.AttributeProto.INTS:
                json_attribute['type'] = 'int64[]'
                json_attribute['value'] = [ item for item in attribute.ints ]
            elif attribute.type == onnx.onnx_pb.AttributeProto.STRINGS:
                json_attribute['type'] = 'string[]'
                json_attribute['value'] = [ item for item in attribute.strings ]
            elif attribute.type == onnx.onnx_pb.AttributeProto.TENSORS:
                raise Exception('Unsupported tensors attribute type')
            elif attribute.type == onnx.onnx_pb.AttributeProto.GRAPHS:
                raise Exception('Unsupported graphs attribute type')
            else:
                raise Exception('Unsupported attribute type')
            json_node['attributes'].append(json_attribute)
        json_graph['nodes'].append(json_node)

    import json
    text = json.dumps(json_model, ensure_ascii=False, indent=2)
    return text.encode('utf-8')

categories = {
    'Constant': 'Constant',
    'Conv': 'Layer',
    'ConvInteger': 'Layer',
    'ConvTranspose': 'Layer',
    'FC': 'Layer',
    'RNN': 'Layer',
    'LSTM': 'Layer',
    'GRU': 'Layer',
    'Gemm': 'Layer',
    'FusedConv': 'Layer',
    'Dropout': 'Dropout',
    'Elu': 'Activation',
    'HardSigmoid': 'Activation',
    'LeakyRelu': 'Activation',
    'PRelu': 'Activation',
    'ThresholdedRelu': 'Activation',
    'Relu': 'Activation',
    'Selu': 'Activation',
    'Sigmoid': 'Activation',
    'Tanh': 'Activation',
    'LogSoftmax': 'Activation',
    'Softmax': 'Activation',
    'Softplus': 'Activation',
    'Softsign': 'Activation',
    'Clip': 'Activation',
    'BatchNormalization': 'Normalization',
    'InstanceNormalization': 'Normalization',
    'LpNormalization': 'Normalization',
    'LRN': 'Normalization',
    'Flatten': 'Shape',
    'Reshape': 'Shape',
    'Tile': 'Shape',
    'Xor': 'Logic',
    'Not': 'Logic',
    'Or': 'Logic',
    'Less': 'Logic',
    'And': 'Logic',
    'Greater': 'Logic',
    'Equal': 'Logic',
    'AveragePool': 'Pool',
    'GlobalAveragePool': 'Pool',
    'GlobalLpPool': 'Pool',
    'GlobalMaxPool': 'Pool',
    'LpPool': 'Pool',
    'MaxPool': 'Pool',
    'MaxRoiPool': 'Pool',
    'Concat': 'Tensor',
    'Slice': 'Tensor',
    'Split': 'Tensor',
    'Pad': 'Tensor',
    'ImageScaler': 'Data',
    'Crop': 'Data',
    'Upsample': 'Data',
    'Transpose': 'Transform',
    'Gather': 'Transform',
    'Unsqueeze': 'Transform',
    'Squeeze': 'Transform',
}

def category(type):
    return categories[type] if type in categories else ''