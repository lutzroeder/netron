
from __future__ import unicode_literals

import onnx
print(onnx.__file__)

import json
import io
import sys

from onnx import defs
from onnx.defs import OpSchema
from onnx.backend.test.case import collect_snippets

snippets = collect_snippets()

categories = {
    'Constant': 'Constant',

    'Conv': 'Layer',
    'ConvTranspose': 'Layer',
    'FC': 'Layer',
    'RNN': 'Layer',
    'LSTM': 'Layer',
    'GRU': 'Layer',
    'Gemm': 'Layer',

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

    'BatchNormalization': 'Normalization',
    'InstanceNormalization': 'Normalization',
    'LpNormalization': 'Normalization',
    'LRN': 'Normalization',

    'Flatten': 'Shape',
    'Reshape': 'Shape',
    'Transpose': 'Shape',
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

    'Gather': 'Transform',
    'Unsqueeze': 'Transform',
    'Squeeze': 'Transform',
}

def generate_json_attr_type(type):
    assert isinstance(type, OpSchema.AttrType)
    s = str(type)
    s = s[s.rfind('.')+1:].lower()
    if s[-1] == 's':
        s = s[0:-1] + '[]'
    return s

def generate_json_attr_default_value(attr_value):
    if not str(attr_value):
        return None
    if attr_value.HasField('i'):
        return attr_value.i
    if attr_value.HasField('s'):
        return attr_value.s.decode('utf8')
    if attr_value.HasField('f'):
        return attr_value.f
    return None

def generate_json_support_level_name(support_level):
    assert isinstance(support_level, OpSchema.SupportType)
    s = str(support_level)
    return s[s.rfind('.')+1:].lower()

def generate_json_types(types):
    r = []
    for type in types:
        r.append(type)
    r = sorted(r)
    return r

def generate_json(schemas, json_file):
    json_root = []
    for schema in schemas:
        json_schema = {}
        if schema.domain:
            json_schema['domain'] = schema.domain
        else:
            json_schema['domain'] = 'ai.onnx'
        json_schema['since_version'] = schema.since_version
        json_schema['support_level'] = generate_json_support_level_name(schema.support_level)
        if schema.doc:
            json_schema['description'] = schema.doc.lstrip()
        if schema.inputs:
            json_schema['inputs'] = []
            for input in schema.inputs:
                json_input = {}
                json_input['name'] = input.name
                json_input['description'] = input.description
                json_input['type'] = input.typeStr
                if input.option == OpSchema.FormalParameterOption.Optional:
                    json_input['option'] = 'optional'
                elif input.option == OpSchema.FormalParameterOption.Variadic:
                    json_input['option'] = 'variadic'
                json_schema['inputs'].append(json_input)
        json_schema['min_input'] = schema.min_input
        json_schema['max_input'] = schema.max_input
        if schema.outputs:
            json_schema['outputs'] = []
            for output in schema.outputs:
                json_output = {}
                json_output['name'] = output.name
                json_output['description'] = output.description
                json_output['type'] = output.typeStr
                if output.option == OpSchema.FormalParameterOption.Optional:
                    json_output['option'] = 'optional'
                elif output.option == OpSchema.FormalParameterOption.Variadic:
                    json_output['option'] = 'variadic'
                json_schema['outputs'].append(json_output)
        json_schema['min_output'] = schema.min_output
        json_schema['max_output'] = schema.max_output
        if schema.attributes:
            json_schema['attributes'] = []
            for _, attribute in sorted(schema.attributes.items()):
                json_attribute = {}
                json_attribute['name'] = attribute.name
                json_attribute['description'] = attribute.description
                json_attribute['type'] = generate_json_attr_type(attribute.type)
                json_attribute['required'] = attribute.required
                default_value = generate_json_attr_default_value(attribute.default_value)
                if default_value:
                    json_attribute['default'] = default_value
                json_schema['attributes'].append(json_attribute)
        if schema.type_constraints:
            json_schema['type_constraints'] = []
            for type_constraint in schema.type_constraints:
                json_schema['type_constraints'].append({
                    'description': type_constraint.description,
                    'type_param_str': type_constraint.type_param_str,
                    'allowed_type_strs': type_constraint.allowed_type_strs
                })
        if schema.name in snippets:
            json_schema['examples'] = []
            for summary, code in sorted(snippets[schema.name]):
                json_schema['examples'].append({
                    'summary': summary,
                    'code': code
                })
        if schema.name in categories:
            json_schema['category'] = categories[schema.name]
        json_root.append({
            'name': schema.name,
            'schema': json_schema 
        })
    with io.open(json_file, 'w', newline='') as fout:
        json_root = json.dumps(json_root, sort_keys=True, indent=2)
        for line in json_root.splitlines():
            line = line.rstrip()
            if sys.version_info[0] < 3:
                line = unicode(line)
            fout.write(line)
            fout.write('\n')

def metadata():
    schemas = defs.get_all_schemas_with_history()
    schemas = sorted(schemas, key=lambda schema: schema.name)
    generate_json(schemas, '../src/onnx-metadata.json')

def convert():
    file = sys.argv[2];
    base, extension = os.path.splitext(file)
    if extension == '.mlmodel':
        import coremltools
        import onnxmltools
        coreml_model = coremltools.utils.load_spec(file)
        onnx_model = onnxmltools.convert.convert_coreml(coreml_model)
        onnxmltools.utils.save_model(onnx_model, base + '.onnx')
    elif extension == '.h5':
        import keras
        import onnxmltools
        keras_model = keras.models.load_model(file)
        onnx_model = onnxmltools.convert.convert_keras(keras_model)
        onnxmltools.utils.save_model(onnx_model, base + '.onnx')
    elif extension == '.pkl':
        from sklearn.externals import joblib
        import onnxmltools
        sklearn_model = joblib.load(file)
        onnx_model = onnxmltools.convert.convert_sklearn(sklearn_model)
        onnxmltools.utils.save_model(onnx_model, base + '.onnx')
    base, extension = os.path.splitext(file)
    if extension == '.onnx':
        import onnx
        from google.protobuf import text_format
        onnx_model = onnx.load(file)
        text = text_format.MessageToString(onnx_model)
        with open(base + '.pbtxt', 'w') as text_file:
            text_file.write(text)

if __name__ == '__main__':
    command_table = { 'metadata': metadata }
    command = sys.argv[1];
    command_table[command]()
