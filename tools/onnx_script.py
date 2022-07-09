''' ONNX metadata script '''

import collections
import io
import json
import os
import re
import sys

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

attribute_type_table = {
    'undefined': None,
    'float': 'float32', 'int': 'int64', 'string': 'string',
    'tensor': 'tensor', 'graph': 'graph',
    'floats': 'float32[]', 'ints': 'int64[]', 'strings': 'string[]',
    'tensors': 'tensor[]', 'graphs': 'graph[]',
}

def _get_attr_type(attribute_type, attribute_name, op_type, op_domain):
    key = op_domain + ':' + op_type + ':' + attribute_name
    if key in (':Cast:to', ':EyeLike:dtype', ':RandomNormal:dtype'):
        return 'DataType'
    value = str(attribute_type)
    value = value[value.rfind('.')+1:].lower()
    if value in attribute_type_table:
        return attribute_type_table[value]
    return None

def _get_attr_default_value(attr_value):
    if not str(attr_value):
        return None
    if attr_value.HasField('i'):
        return attr_value.i
    if attr_value.HasField('s'):
        return attr_value.s.decode('utf8')
    if attr_value.HasField('f'):
        return attr_value.f
    return None

def _generate_json_support_level_name(support_level):
    value = str(support_level)
    return value[value.rfind('.')+1:].lower()

def _format_description(description):
    def replace_line(match):
        link = match.group(1)
        url = match.group(2)
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://github.com/onnx/onnx/blob/master/docs/" + url
        return "[" + link + "](" + url + ")"
    description = re.sub("\\[(.+)\\]\\(([^ ]+?)( \"(.+)\")?\\)", replace_line, description)
    return description

def _metadata():
    json_root = []
    import onnx.backend.test.case # pylint: disable=import-outside-toplevel
    import onnx.defs # pylint: disable=import-outside-toplevel
    snippets = onnx.backend.test.case.collect_snippets()
    all_schemas_with_history = onnx.defs.get_all_schemas_with_history()
    for schema in all_schemas_with_history:
        json_schema = {}
        json_schema['name'] = schema.name
        json_schema['module'] = schema.domain if schema.domain else 'ai.onnx'
        json_schema['version'] = schema.since_version
        json_schema['support_level'] = _generate_json_support_level_name(schema.support_level)
        if schema.doc:
            json_schema['description'] = _format_description(schema.doc.lstrip())
        if schema.attributes:
            json_schema['attributes'] = []
            for _ in collections.OrderedDict(schema.attributes.items()).values():
                json_attribute = {}
                json_attribute['name'] = _.name
                attribute_type = _get_attr_type(_.type, _.name, schema.name, schema.domain)
                if attribute_type:
                    json_attribute['type'] = attribute_type
                elif 'type' in json_attribute:
                    del json_attribute['type']
                json_attribute['required'] = _.required
                default_value = _get_attr_default_value(_.default_value)
                if default_value:
                    json_attribute['default'] = default_value
                json_attribute['description'] = _format_description(_.description)
                json_schema['attributes'].append(json_attribute)
        if schema.inputs:
            json_schema['inputs'] = []
            for _ in schema.inputs:
                json_input = {}
                json_input['name'] = _.name
                json_input['type'] = _.typeStr
                if _.option == onnx.defs.OpSchema.FormalParameterOption.Optional:
                    json_input['option'] = 'optional'
                elif _.option == onnx.defs.OpSchema.FormalParameterOption.Variadic:
                    json_input['list'] = True
                json_input['description'] = _format_description(_.description)
                json_schema['inputs'].append(json_input)
        json_schema['min_input'] = schema.min_input
        json_schema['max_input'] = schema.max_input
        if schema.outputs:
            json_schema['outputs'] = []
            for _ in schema.outputs:
                json_output = {}
                json_output['name'] = _.name
                json_output['type'] = _.typeStr
                if _.option == onnx.defs.OpSchema.FormalParameterOption.Optional:
                    json_output['option'] = 'optional'
                elif _.option == onnx.defs.OpSchema.FormalParameterOption.Variadic:
                    json_output['list'] = True
                json_output['description'] = _format_description(_.description)
                json_schema['outputs'].append(json_output)
        json_schema['min_output'] = schema.min_output
        json_schema['max_output'] = schema.max_output
        def format_range(value):
            return '&#8734;' if value == 2147483647 else str(value)
        if schema.min_input != schema.max_input:
            json_schema['inputs_range'] = format_range(schema.min_input) + ' - ' + format_range(schema.max_input)
        if schema.min_output != schema.max_output:
            json_schema['outputs_range'] = format_range(schema.min_output) + ' - ' + format_range(schema.max_output)
        if schema.type_constraints:
            json_schema['type_constraints'] = []
            for type_constraint in schema.type_constraints:
                json_schema['type_constraints'].append({
                    'description': type_constraint.description,
                    'type_param_str': type_constraint.type_param_str,
                    'allowed_type_strs': type_constraint.allowed_type_strs
                })
        if schema.name in snippets:
            def update_code(code):
                lines = code.splitlines()
                while len(lines) > 0 and re.search("\\s*#", lines[-1]):
                    lines.pop()
                    if len(lines) > 0 and len(lines[-1]) == 0:
                        lines.pop()
                return '\n'.join(lines)
            json_schema['examples'] = []
            for summary, code in sorted(snippets[schema.name]):
                json_schema['examples'].append({
                    'summary': summary,
                    'code': update_code(code)
                })
        if schema.name in categories:
            json_schema['category'] = categories[schema.name]
        json_root.append(json_schema)
    json_root = sorted(json_root, key=lambda item: item['name'] + ':' + str(item['version'] if 'version' in item else 0).zfill(4))
    json_file = os.path.join(os.path.dirname(__file__), '../source/onnx-metadata.json')
    with io.open(json_file, 'r', encoding='utf-8') as file:
        content = file.read()
        items = json.loads(content)
        items = list(filter(lambda item: item['module'] == "com.microsoft", items))
        json_root = json_root + items
    json_root = json.dumps(json_root, indent=2)
    with io.open(json_file, 'w', encoding='utf-8', newline='') as file:
        for line in json_root.splitlines():
            file.write(line.rstrip() + '\n')

def _infer():
    import onnx # pylint: disable=import-outside-toplevel
    import onnx.shape_inference # pylint: disable=import-outside-toplevel
    file = sys.argv[2]
    base = os.path.splitext(file)[0]
    onnx_model = onnx.load(base + '.onnx')
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, base + '.shape.onnx')

def main(): # pylint: disable=missing-function-docstring
    command_table = { 'metadata': _metadata, 'infer': _infer }
    command = sys.argv[1]
    command_table[command]()

if __name__ == '__main__':
    main()
