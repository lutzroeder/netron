''' ONNX backend '''

import collections
import enum
import json
import os

class ModelFactory: # pylint: disable=too-few-public-methods
    ''' ONNX backend model factory '''
    def open(self, model): # pylint: disable=missing-function-docstring
        return _Model(model)

class _Model: # pylint: disable=too-few-public-methods
    def __init__(self, model):
        ''' Serialize ONNX model to JSON message '''
        # import onnx.shape_inference
        # model = onnx.shape_inference.infer_shapes(model)
        self.value = model
        self.metadata = _Metadata()
        self.graph = _Graph(model.graph, self.metadata)

    def to_json(self): # pylint: disable=missing-function-docstring
        ''' Serialize model to JSON message '''
        model = self.value
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
        json_metadata = self._metadata_props(model.metadata_props)
        if len(json_metadata) > 0:
            json_model['metadata'] = json_metadata
        json_model['graphs'] = []
        json_model['graphs'].append(self.graph.to_json())
        return json_model

    def _metadata_props(self, metadata_props): # pylint: disable=missing-function-docstring
        json_metadata = []
        metadata_props = [ [ entry.key, entry.value ] for entry in metadata_props ]
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
        return json_metadata

class _Graph:
    def __init__(self, graph, metadata):
        self.metadata = metadata
        self.value = graph
        self.arguments_index = {}
        self.arguments = []

    def _tensor(self, tensor): # pylint: disable=unused-argument
        return {}

    def argument(self, name, tensor_type=None, initializer=None): # pylint: disable=missing-function-docstring
        if not name in self.arguments_index:
            argument = _Argument(name, tensor_type, initializer)
            self.arguments_index[name] = len(self.arguments)
            self.arguments.append(argument)
        index = self.arguments_index[name]
        # argument.set_initializer(initializer)
        return index

    def attribute(self, _, op_type): # pylint: disable=missing-function-docstring,too-many-branches
        if _.type == _AttributeType.UNDEFINED:
            attribute_type = None
            value = None
        elif _.type == _AttributeType.FLOAT:
            attribute_type = 'float32'
            value = _.f
        elif _.type == _AttributeType.INT:
            attribute_type = 'int64'
            value = _.i
        elif _.type == _AttributeType.STRING:
            attribute_type = 'string'
            value = _.s.decode('latin1' if op_type == 'Int8GivenTensorFill' else 'utf-8')
        elif _.type == _AttributeType.TENSOR:
            attribute_type = 'tensor'
            value = self._tensor(_.t)
        elif _.type == _AttributeType.GRAPH:
            attribute_type = 'tensor'
            raise Exception('Unsupported graph attribute type') # pylint: disable=broad-exception-raised
        elif _.type == _AttributeType.FLOATS:
            attribute_type = 'float32[]'
            value = list(_.floats)
        elif _.type == _AttributeType.INTS:
            attribute_type = 'int64[]'
            value = list(_.ints)
        elif _.type == _AttributeType.STRINGS:
            attribute_type = 'string[]'
            value = [ item.decode('utf-8') for item in _.strings ]
        elif _.type == _AttributeType.TENSORS:
            attribute_type = 'tensor[]'
            raise Exception('Unsupported tensors attribute type') # pylint: disable=broad-exception-raised
        elif _.type == _AttributeType.GRAPHS:
            attribute_type = 'graph[]'
            raise Exception('Unsupported graphs attribute type') # pylint: disable=broad-exception-raised
        elif _.type == _AttributeType.SPARSE_TENSOR:
            attribute_type = 'tensor'
            value = self._tensor(_.sparse_tensor)
        else:
            raise Exception("Unsupported attribute type '" + str(_.type) + "'.") # pylint: disable=broad-exception-raised
        json_attribute = {}
        json_attribute['name'] = _.name
        if attribute_type:
            json_attribute['type'] = attribute_type
        json_attribute['value'] = value
        return json_attribute

    def to_json(self): # pylint: disable=missing-function-docstring
        graph = self.value
        json_graph = {
            'nodes': [],
            'inputs': [],
            'outputs': [],
            'arguments': []
        }
        for value_info in graph.value_info:
            self.argument(value_info.name)
        for initializer in graph.initializer:
            self.argument(initializer.name, None, initializer)
        for node in graph.node:
            op_type = node.op_type
            json_node = {}
            json_node_type = {}
            json_node_type['name'] = op_type
            type_metadata = self.metadata.type(op_type)
            if type and 'category' in type_metadata:
                json_node_type['category'] = type_metadata['category']
            json_node['type'] = json_node_type
            if node.name:
                json_node['name'] = node.name
            json_node['inputs'] = []
            for value in node.input:
                json_node['inputs'].append({
                        'name': 'X',
                        'arguments': [ self.argument(value) ]
                    })
            json_node['outputs'] = []
            for value in node.output:
                json_node['outputs'].append({
                        'name': 'X',
                        'arguments': [ self.argument(value) ]
                    })
            json_node['attributes'] = []
            for _ in node.attribute:
                json_attribute = self.attribute(_, op_type)
                json_node['attributes'].append(json_attribute)
            json_graph['nodes'].append(json_node)
        for _ in self.arguments:
            json_graph['arguments'].append(_.to_json())
        return json_graph

class _Argument: # pylint: disable=too-few-public-methods
    def __init__(self, name, tensor_type=None, initializer=None):
        self.name = name
        self.type = tensor_type
        self.initializer = initializer

    def to_json(self): # pylint: disable=missing-function-docstring
        target = {}
        target['name'] = self.name
        if self.initializer:
            target['initializer'] = {}
        return target

class _Metadata: # pylint: disable=too-few-public-methods
    metadata = {}

    def __init__(self):
        metadata_file = os.path.join(os.path.dirname(__file__), 'onnx-metadata.json')
        with open(metadata_file, 'r', encoding='utf-8') as file:
            for item in json.load(file):
                name = item['name']
                self.metadata[name] = item

    def type(self, name): # pylint: disable=missing-function-docstring
        if name in self.metadata:
            return self.metadata[name]
        return {}

class _AttributeType(enum.IntEnum):
    UNDEFINED = 0
    FLOAT = 1
    INT = 2
    STRING = 3
    TENSOR = 4
    GRAPH = 5
    FLOATS = 6
    INTS = 7
    STRINGS = 8
    TENSORS = 9
    GRAPHS = 10
    SPARSE_TENSOR = 11
    SPARSE_TENSORS = 12
    TYPE_PROTO = 13
    TYPE_PROTOS = 14
