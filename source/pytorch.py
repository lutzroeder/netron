''' PyTorch backend '''

import json
import os

class ModelFactory:
    ''' PyTorch backend model factory '''
    def serialize(self, model):
        ''' Serialize PyTorch model to JSON message '''
        import torch # pylint: disable=import-outside-toplevel
        metadata = {}
        metadata_file = os.path.join(os.path.dirname(__file__), 'onnx-metadata.json')
        with open(metadata_file, 'r', encoding='utf-8') as file:
            for item in json.load(file):
                name = 'onnx::' + item['name']
                metadata[name] = item

        json_model = {}
        json_model['signature'] = 'netron:pytorch'
        json_model['format']  = 'TorchScript'
        json_model['graphs'] = []
        json_graph = {}
        json_graph['arguments'] = []
        json_graph['nodes'] = []
        json_graph['inputs'] = []
        json_graph['outputs'] = []
        json_model['graphs'].append(json_graph)
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

        for input_value in model.inputs():
            json_graph['inputs'].append({
                'name': input_value.debugName(),
                'arguments': [ argument(input_value) ]
            })
        for output_value in model.outputs():
            json_graph['outputs'].append({
                'name': output_value.debugName(),
                'arguments': [ argument(output_value) ]
            })
        for node in model.nodes():
            kind = node.kind()
            json_type = {
                'name': kind
            }
            json_node = {
                'type': json_type,
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

        text = json.dumps(json_model, ensure_ascii=False)
        return text.encode('utf-8')
