''' ONNX backend '''

import collections
import json

class ModelFactory:
    ''' ONNX backend model factory '''
    def serialize(self, model):
        ''' Serialize ONNX model to JSON message '''
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
        value = metadata.get('converted_from')
        if value:
            json_metadata.append({ 'name': 'source', 'value': value })
        value = metadata.get('author')
        if value:
            json_metadata.append({ 'name': 'author', 'value': value })
        value = metadata.get('company')
        if value:
            json_metadata.append({ 'name': 'company', 'value': value })
        value = metadata.get('license')
        license_url = metadata.get('license_url')
        if license_url:
            value = '<a href=\'' + license_url + '\'>' + (value if value else license_url) + '</a>'
        if value:
            json_metadata.append({ 'name': 'license', 'value': value })
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
        arguments = {}
        def tensor(tensor):
            return {}
        def argument(name, tensor_type=None, initializer=None):
            if not name in arguments:
                json_argument = {}
                json_argument['name'] = name
                arguments[name] = len(json_graph['arguments'])
                json_graph['arguments'].append(json_argument)
            index = arguments[name]
            if tensor_type or initializer:
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
            if self.category(op_type):
                json_node_type['category'] = self.category(op_type)
            json_node['type'] = json_node_type
            if node.name:
                json_node['name'] = node.name
            json_node['inputs'] = []
            for value in node.input:
                json_node['inputs'].append({
                        'name': 'X',
                        'arguments': [ argument(value) ]
                    })
            json_node['outputs'] = []
            for value in node.output:
                json_node['outputs'].append({
                        'name': 'X',
                        'arguments': [ argument(value) ]
                    })
            json_node['attributes'] = []
            for _ in node.attribute:
                if _.type == onnx.onnx_pb.AttributeProto.UNDEFINED:
                    attribute_type = None
                    value = None
                elif _.type == onnx.onnx_pb.AttributeProto.FLOAT:
                    attribute_type = 'float32'
                    value = _.f
                elif _.type == onnx.onnx_pb.AttributeProto.INT:
                    attribute_type = 'int64'
                    value = _.i
                elif _.type == onnx.onnx_pb.AttributeProto.STRING:
                    attribute_type = 'string'
                    value = _.s.decode('latin1' if op_type == 'Int8GivenTensorFill' else 'utf-8')
                elif _.type == onnx.onnx_pb.AttributeProto.TENSOR:
                    attribute_type = 'tensor'
                    value = tensor(_.t)
                elif _.type == onnx.onnx_pb.AttributeProto.GRAPH:
                    attribute_type = 'tensor'
                    raise Exception('Unsupported graph attribute type')
                elif _.type == onnx.onnx_pb.AttributeProto.FLOATS:
                    attribute_type = 'float32[]'
                    value = [ item for item in _.floats ]
                elif _.type == onnx.onnx_pb.AttributeProto.INTS:
                    attribute_type = 'int64[]'
                    value = [ item for item in _.ints ]
                elif _.type == onnx.onnx_pb.AttributeProto.STRINGS:
                    attribute_type = 'string[]'
                    value = [ item.decode('utf-8') for item in _.strings ]
                elif _.type == onnx.onnx_pb.AttributeProto.TENSORS:
                    attribute_type = 'tensor[]'
                    raise Exception('Unsupported tensors attribute type')
                elif _.type == onnx.onnx_pb.AttributeProto.GRAPHS:
                    attribute_type = 'graph[]'
                    raise Exception('Unsupported graphs attribute type')
                elif _.type == onnx.onnx_pb.AttributeProto.SPARSE_TENSOR:
                    attribute_type = 'tensor'
                    value = tensor(_.sparse_tensor)
                else:
                    raise Exception("Unsupported attribute type '" + str(_.type) + "'.")
                json_attribute = {}
                json_attribute['name'] = _.name
                if attribute_type:
                    json_attribute['type'] = attribute_type
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

    def category(self, name):
        ''' Get category for type '''
        return self.categories[name] if name in self.categories else ''
