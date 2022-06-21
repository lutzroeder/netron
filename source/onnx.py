import collections
import json

def serialize(model):
    print('Experimental')
    # import onnx.shape_inference
    # model = onnx.shape_inference.infer_shapes(model)
    import onnx.onnx_pb # pylint: disable=import-outside-toplevel
    json_model = {}
    json_model['signature'] = 'netron:onnx'
    json_model['format'] = 'ONNX' + (' v' + str(model.ir_version) if model.ir_version else '')
    if model.producer_name and len(model.producer_name) > 0:
        producer_version = ' v' + model.producer_version if model.producer_version else ''
        json_model['producer'] = model.producer_name + producer_version
    if model.model_version and model.model_version != 0:
        json_model['version'] = str(model.model_version)
    if model.doc_string and len(model.doc_string):
        json_model['description'] = str(model.doc_string)
    json_metadata = []
    metadata_props = [ [ entry.key, entry.value ] for entry in model.metadata_props ]
    metadata = collections.OrderedDict(metadata_props)
    converted_from = metadata.get('converted_from')
    if converted_from:
        json_metadata.append({ 'name': 'source', 'value': converted_from })
    author = metadata.get('author')
    if author:
        json_metadata.append({ 'name': 'author', 'value': author })
    company = metadata.get('company')
    if company:
        json_metadata.append({ 'name': 'company', 'value': company })
    license = metadata.get('license')
    license_url = metadata.get('license_url')
    if license_url:
        license = '<a href=\'' + license_url + '\'>' + (license if license else license_url) + '</a>'
    if license:
        json_metadata.append({ 'name': 'license', 'value': license })
    if 'author' in metadata:
        metadata.pop('author')
    if 'company' in metadata:
        metadata.pop('company')
    if 'converted_from' in metadata:
        metadata.pop('converted_from')
    if 'license' in metadata:
        metadata.pop('license')
    if 'license_url' in metadata:
        metadata.pop('license_url')
    for name, value in metadata.items():
        json_metadata.append({ 'name': name, 'value': value })
    if len(json_metadata) > 0:
        json_model['metadata'] = json_metadata
    json_model['graphs'] = []
    graph = model.graph
    json_graph = {
        'nodes': [],
        'inputs': [],
        'outputs': [],
        'arguments': []
    }
    json_model['graphs'].append(json_graph)
    arguments = dict()
    def tensor(tensor):
        return {}
    def argument(name, type=None, initializer=None):
        if not name in arguments:
            json_argument = {}
            json_argument['name'] = name
            arguments[name] = len(json_graph['arguments'])
            json_graph['arguments'].append(json_argument)
        index = arguments[name]
        if type or initializer:
            json_argument = json_graph['arguments'][index]
            if initializer:
                json_argument['initializer'] = tensor(initializer)
        return index

    for value_info in graph.value_info:
        argument(value_info.name)
    for initializer in graph.initializer:
        argument(initializer.name, None, initializer)
    for node in graph.node:
        op_type = node.op_type
        json_node = {}
        json_node_type = {}
        json_node_type['name'] = op_type
        if category(op_type):
            json_node_type['category'] = category(op_type)
        json_node['type'] = json_node_type
        if node.name:
            json_node['name'] = node.name
        json_node['inputs'] = []
        for input in node.input:
            json_node['inputs'].append({
                    'name': 'X',
                    'arguments': [ argument(input) ]
                })
        json_node['outputs'] = []
        for output in node.output:
            json_node['outputs'].append({
                    'name': 'X',
                    'arguments': [ argument(output) ]
                })
        json_node['attributes'] = []
        for attribute in node.attribute:
            if attribute.type == onnx.onnx_pb.AttributeProto.UNDEFINED:
                type = None
                value = None
            elif attribute.type == onnx.onnx_pb.AttributeProto.FLOAT:
                type = 'float32'
                value = attribute.f
            elif attribute.type == onnx.onnx_pb.AttributeProto.INT:
                type = 'int64'
                value = attribute.i
            elif attribute.type == onnx.onnx_pb.AttributeProto.STRING:
                type = 'string'
                value = attribute.s.decode('latin1') if op_type == 'Int8GivenTensorFill' else attribute.s.decode('utf-8')
            elif attribute.type == onnx.onnx_pb.AttributeProto.TENSOR:
                type = 'tensor'
                value = tensor(attribute.t)
            elif attribute.type == onnx.onnx_pb.AttributeProto.GRAPH:
                graph = 'tensor'
                raise Exception('Unsupported graph attribute type')
            elif attribute.type == onnx.onnx_pb.AttributeProto.FLOATS:
                type = 'float32[]'
                value = [ item for item in attribute.floats ]
            elif attribute.type == onnx.onnx_pb.AttributeProto.INTS:
                type = 'int64[]'
                value = [ item for item in attribute.ints ]
            elif attribute.type == onnx.onnx_pb.AttributeProto.STRINGS:
                type = 'string[]'
                value = [ item.decode('utf-8') for item in attribute.strings ]
            elif attribute.type == onnx.onnx_pb.AttributeProto.TENSORS:
                type = 'tensor[]'
                raise Exception('Unsupported tensors attribute type')
            elif attribute.type == onnx.onnx_pb.AttributeProto.GRAPHS:
                type = 'graph[]'
                raise Exception('Unsupported graphs attribute type')
            elif attribute.type == onnx.onnx_pb.AttributeProto.SPARSE_TENSOR:
                type = 'tensor'
                value = tensor(attribute.sparse_tensor)
            else:
                raise Exception("Unsupported attribute type '" + str(attribute.type) + "'.")
            json_attribute = {}
            json_attribute['name'] = attribute.name
            if type:
                json_attribute['type'] = type
            json_attribute['value'] = value
            json_node['attributes'].append(json_attribute)
        json_graph['nodes'].append(json_node)
    text = json.dumps(json_model, ensure_ascii=False)
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

def category(name):
    return categories[name] if name in categories else ''
