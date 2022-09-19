''' PyTorch backend '''

import json
import os

class ModelFactory: # pylint: disable=too-few-public-methods
    ''' PyTorch backend model factory '''
    def open(self, model): # pylint: disable=missing-function-docstring
        return _Model(model)

class _Model: # pylint: disable=too-few-public-methods
    def __init__(self, model):
        self.graph = _Graph(model)

    def to_json(self):
        ''' Serialize model to JSON message '''
        metadata = {}
        metadata_file = os.path.join(os.path.dirname(__file__), 'onnx-metadata.json')
        with open(metadata_file, 'r', encoding='utf-8') as file:
            for item in json.load(file):
                name = 'onnx::' + item['name']
                metadata[name] = item
        json_model = {
            'signature': 'netron:pytorch',
            'format': 'TorchScript',
            'graphs': [ self.graph.to_json() ]
        }
        return json_model

class _Graph: # pylint: disable=too-few-public-methods

    def __init__(self, graph):
        self.value = graph

    def to_json(self): # pylint: disable=missing-function-docstring
        import torch # pylint: disable=import-outside-toplevel
        graph = self.value
        json_graph = {
            'arguments': [],
            'nodes': [],
            'inputs': [],
            'outputs': []
        }
        data_type_map = dict([
            [ torch.float16, 'float16'], # pylint: disable=no-member
            [ torch.float32, 'float32'], # pylint: disable=no-member
            [ torch.float64, 'float64'], # pylint: disable=no-member
            [ torch.int32, 'int32'], # pylint: disable=no-member
            [ torch.int64, 'int64'], # pylint: disable=no-member
        ])
        arguments_map = {}
        def argument(value):
            if not value in arguments_map:
                json_argument = {}
                json_argument['name'] = str(value.unique()) + '>' + str(value.node().kind())
                if value.isCompleteTensor():
                    json_tensor_shape = {
                        'dimensions': value.type().sizes()
                    }
                    json_argument['type'] = {
                        'dataType': data_type_map[value.type().dtype()],
                        'shape': json_tensor_shape
                    }
                if value.node().kind() == "prim::Param":
                    json_argument['initializer'] = {}
                arguments = json_graph['arguments']
                arguments_map[value] = len(arguments)
                arguments.append(json_argument)
            return arguments_map[value]

        for _ in graph.inputs():
            json_graph['inputs'].append({
                'name': _.debugName(),
                'arguments': [ argument(_) ]
            })
        for _ in graph.outputs():
            json_graph['outputs'].append({
                'name': _.debugName(),
                'arguments': [ argument(_) ]
            })
        for node in graph.nodes():
            json_node = {
                'type': { 'name': node.kind() },
                'inputs': [],
                'outputs': [],
                'attributes': []
            }
            json_graph['nodes'].append(json_node)
            for name in node.attributeNames():
                value = node[name]
                json_attribute = {
                    'name': name,
                    'value': value
                }
                if torch.is_tensor(value):
                    json_node['inputs'].append({
                        'name': name,
                        'arguments': []
                    })
                else:
                    json_node['attributes'].append(json_attribute)

            for input_value in node.inputs():
                json_parameter = {
                    'name': 'x',
                    'arguments': [ argument(input_value) ]
                }
                json_node['inputs'].append(json_parameter)

            for output_value in node.outputs():
                json_node['outputs'].append({
                    'name': 'x',
                    'arguments': [ argument(output_value) ]
                })
        return json_graph
