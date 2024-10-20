''' PyTorch backend '''

import json
import os

class ModelFactory: # pylint: disable=too-few-public-methods
    ''' PyTorch backend model factory '''
    def open(self, model): # pylint: disable=missing-function-docstring
        metadata = {}
        metadata_files = [
            ('pytorch-metadata.json', ''),
            ('onnx-metadata.json', 'onnx::')
        ]
        path = os.path.dirname(__file__)
        for entry in metadata_files:
            file = os.path.join(path, entry[0])
            with open(file, 'r', encoding='utf-8') as handle:
                for item in json.load(handle):
                    name = entry[1] + item['name'].split('(', 1)[0]
                    metadata[name] = item
        metadata = Metadata(metadata)
        return _Model(metadata, model)

class _Model: # pylint: disable=too-few-public-methods
    def __init__(self, metadata, model):
        self.graph = _Graph(metadata, model)

    def to_json(self):
        ''' Serialize model to JSON message '''
        import torch # pylint: disable=import-outside-toplevel,import-error
        json_model = {
            'signature': 'netron:pytorch',
            'format': 'TorchScript v' + torch.__version__,
            'graphs': [ self.graph.to_json() ]
        }
        return json_model

class _Graph: # pylint: disable=too-few-public-methods

    def __init__(self, metadata, model):
        self.metadata = metadata
        self.param = model
        self.value = model.graph
        self.nodes = []

    def _getattr(self, node):
        if node.kind() == 'prim::Param':
            return (self.param, '')
        if node.kind() == 'prim::GetAttr':
            name = node.s('name')
            obj, parent = self._getattr(node.input().node())
            return (getattr(obj, name), parent + '.' + name if len(parent) > 0 else name)
        raise NotImplementedError()

    def to_json(self): # pylint: disable=missing-function-docstring,too-many-locals,too-many-statements,too-many-branches
        import torch # pylint: disable=import-outside-toplevel,import-error
        graph = self.value
        json_graph = {
            'values': [],
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
        def constant_value(node):
            if node.hasAttribute('value'):
                selector = node.kindOf('value')
                return getattr(node, selector)('value')
            return None
        values_index = {}
        def argument(value):
            if not value in values_index:
                json_value = {}
                json_value['name'] = str(value.unique())
                node = value.node()
                if node.kind() == "prim::GetAttr":
                    tensor, name = self._getattr(node)
                    if tensor is not None and len(name) > 0 and \
                        isinstance(tensor, torch.Tensor):
                        json_tensor_shape = {
                            'dimensions': list(tensor.shape)
                        }
                        tensor_type = {
                            'dataType': data_type_map[tensor.dtype],
                            'shape': json_tensor_shape
                        }
                        json_value['name'] = name
                        json_value['type'] = tensor_type
                        json_value['initializer'] = { 'type': tensor_type }
                elif node.kind() == "prim::Constant":
                    tensor = constant_value(node)
                    if tensor and isinstance(tensor, torch.Tensor):
                        json_tensor_shape = {
                            'dimensions': list(tensor.shape)
                        }
                        tensor_type = {
                            'dataType': data_type_map[tensor.dtype],
                            'shape': json_tensor_shape
                        }
                        json_value['type'] = tensor_type
                        json_value['initializer'] = { 'type': tensor_type }
                elif value.isCompleteTensor():
                    json_tensor_shape = {
                        'dimensions': value.type().sizes()
                    }
                    json_value['type'] = {
                        'dataType': data_type_map[value.type().dtype()],
                        'shape': json_tensor_shape
                    }
                values = json_graph['values']
                values_index[value] = len(values)
                values.append(json_value)
            return values_index[value]

        for value in graph.inputs():
            if len(value.uses()) != 0 and value.type().kind() != 'ClassType':
                json_graph['inputs'].append({
                    'name': value.debugName(),
                    'value': [ argument(value) ]
                })
        for value in graph.outputs():
            json_graph['outputs'].append({
                'name': value.debugName(),
                'value': [ argument(value) ]
            })
        constants = {}
        for node in graph.nodes():
            if node.kind() == 'prim::Constant':
                constants[node] = 0

        lists = {}
        for node in graph.nodes():
            if node.kind() == 'prim::ListConstruct':
                if all(_.node() in constants for _ in node.inputs()):
                    for _ in node.inputs():
                        constants[_.node()] += 1
                    lists[node] = 0

        def create_node(node):
            identifier = node.schema()
            schema, category = self.metadata.type(identifier)
            json_node = {
                'type': {
                    'name': node.kind(),
                    'category': category
                },
                'inputs': [],
                'outputs': [],
                'attributes': []
            }
            json_graph['nodes'].append(json_node)
            for name in node.attributeNames():
                selector = node.kindOf(name)
                value = getattr(node, selector)(name)
                json_attribute = {
                    'name': name,
                    'value': value
                }
                if torch.is_tensor(value):
                    json_node['inputs'].append({
                        'name': name,
                        'value': []
                    })
                else:
                    json_node['attributes'].append(json_attribute)

            for i, value in enumerate(node.inputs()):
                arg = schema.arguments[i] if schema and i < len(schema.arguments) else None
                parameter_name = arg.name if arg else 'input'
                real_type = arg.real_type if arg else None
                input_node = value.node()
                if input_node in constants:
                    if (real_type and real_type.kind() == 'TensorType') or \
                        value.type().kind() == 'TensorType':
                        json_node['inputs'].append({
                            'name': parameter_name,
                            'value': [ argument(value) ]
                        })
                    else:
                        json_attribute = {
                            'name': parameter_name,
                            'value': constant_value(input_node)
                        }
                        if real_type:
                            json_attribute['type'] = self._argument_type(real_type)
                        json_node['attributes'].append(json_attribute)
                    constants[input_node] = constants[input_node] + 1
                    continue
                if input_node in lists:
                    json_attribute = {
                        'name': parameter_name,
                        'value': [ constant_value(_.node()) for _ in input_node.inputs() ]
                    }
                    json_node['attributes'].append(json_attribute)
                    lists[input_node] += 1
                    continue
                if input_node.kind() == 'prim::TupleUnpack':
                    continue
                if input_node.kind() == 'prim::TupleConstruct':
                    continue
                json_node['inputs'].append({
                    'name': parameter_name,
                    'value': [ argument(value) ]
                })

            for i, value in enumerate(node.outputs()):
                ret = schema.returns[i] if schema and i < len(schema.returns) else None
                name = ret.name if ret else 'output'
                json_node['outputs'].append({
                    'name': name,
                    'value': [ argument(value) ]
                })

        for node in graph.nodes():
            if node in lists:
                continue
            if node in constants:
                continue
            if node.kind() == 'prim::GetAttr':
                continue
            create_node(node)

        for node in graph.nodes():
            if node.kind() == 'prim::Constant' and \
                node in constants and constants[node] != len(node.output().uses()):
                create_node(node)
            if node.kind() == 'prim::ListConstruct' and \
                node in lists and lists[node] != len(node.output().uses()):
                create_node(node)

        return json_graph

    def _argument_type(self, value): # pylint: disable=too-many-branches,too-many-return-statements
        if value.kind() == 'TensorType':
            return 'Tensor'
        if value.kind() == 'OptionalType':
            element_type = self._argument_type(value.getElementType())
            return f'{element_type}?'
        if value.kind() == 'ListType':
            element_type = self._argument_type(value.getElementType())
            size = str(value.size) if hasattr(value, 'size') else ''
            return f'{element_type}[{size}]'
        if value.kind() == 'DictType':
            key_type = self._argument_type(value.getKeyType())
            value_type = self._argument_type(value.getValueType())
            return f'Dict({key_type}, {value_type})'
        if value.kind() == 'TupleType':
            elements = []
            for element in value.elements():
                elements.append(self._argument_type(element))
            return f'({', '.join(elements)})'
        if value.kind() == 'IntType':
            return 'int64'
        if value.kind() == 'SymIntType':
            return 'SymInt'
        if value.kind() == 'FloatType':
            return 'float32'
        if value.kind() == 'BoolType':
            return 'boolean'
        if value.kind() == 'StringType':
            return 'string'
        if value.kind() == 'NumberType':
            return 'Scalar'
        if value.kind() == 'ScalarTypeType':
            return 'ScalarType'
        if value.kind() == 'LayoutType':
            return 'Layout'
        if value.kind() == 'MemoryFormatType':
            return 'MemoryFormat'
        if value.kind() == 'DeviceObjType':
            return 'Device'
        if value.kind() == 'GeneratorType':
            return 'Generator'
        if value.kind() == 'VarType':
            return value.annotation_str
        raise NotImplementedError()

class Metadata: # pylint: disable=too-few-public-methods,missing-class-docstring

    def __init__(self, metadata):
        self.types = metadata

    def type(self, identifier): # pylint: disable=missing-function-docstring
        if identifier == '(no schema)':
            return (None, '')
        key = identifier.split('(', 1)[0]
        value = self.types.get(key)
        category = value['category'] if value and 'category' in value else ''
        name, overload_name = key.split('.', 1) if key.find('.') > 0 else (key, '')
        import torch # pylint: disable=import-outside-toplevel,import-error
        schema = torch._C._get_schema(name, overload_name) # pylint: disable=protected-access
        return (schema, category)
