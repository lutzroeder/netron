''' NNabla metadata script '''

import json
import sys
import os
import yaml # pylint: disable=import-error
import mako # pylint: disable=import-error
import mako.template # pylint: disable=import-error

def _write(path, content):
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)

def _read_yaml(path):
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def _generate_from_template(template_path, **kwargs):
    path = template_path.replace('.tmpl', '')
    template = mako.template.Template(text=None, filename=template_path, preprocessor=None)
    content = template.render(**kwargs)
    content = content.replace('\r\n', '\n')
    content = content.replace('\r', '\n')
    _write(path, content)

def _metadata():
    def parse_functions(function_info):
        functions = []
        for category_name, category in function_info.items():
            for function_name, function_value in category.items():
                function = {
                    'name': function_name,
                    'description': function_value['doc'].strip()
                }
                for input_name, input_value in function_value.get('inputs', {}).items():
                    function.setdefault('inputs', []).append({
                        'name': input_name,
                        'type': 'nnabla.Variable',
                        'option': 'optional' if input_value.get('optional', False) else None,
                        'list': input_value.get('variadic', False),
                        'description': input_value['doc'].strip()
                    })
                for arg_name, arg_value in function_value.get('arguments', {}).items():
                    function.setdefault('attributes', []).append({
                        'name': arg_name,
                        'type': arg_value['type'],
                        'required': 'default' not in arg_value,
                        'default': _try_eval_default(arg_value.get('default', None)),
                        'description': arg_value['doc'].strip()
                    })
                for output_name, output_value in function_value.get('outputs', {}).items():
                    function.setdefault('outputs', []).append({
                        'name': output_name,
                        'type': 'nnabla.Variable',
                        'list': output_value.get('variadic', False),
                        'description': output_value['doc'].strip()
                    })
                if 'Pooling' in function_name:
                    function['category'] = 'Pool'
                elif category_name == 'Neural Network Layer':
                    function['category'] = 'Layer'
                elif category_name == 'Neural Network Activation Functions':
                    function['category'] = 'Activation'
                elif category_name == 'Normalization':
                    function['category'] = 'Normalization'
                elif category_name == 'Logical':
                    function['category'] = 'Logic'
                elif category_name == 'Array Manipulation':
                    function['category'] = 'Shape'
                functions.append(function)
        return functions
    def cleanup_functions(functions):
        for function in functions:
            for inp in function.get('inputs', []):
                if inp['option'] is None:
                    inp.pop('option', None)
                if not inp['list']:
                    inp.pop('list', None)
            for attribute in function.get('attributes', []):
                if attribute['required']:
                    attribute.pop('default', None)
            for output in function.get('outputs', []):
                if not output['list']:
                    output.pop('list', None)
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    functions_yaml_path = os.path.join(root, \
        'third_party', 'source', 'nnabla', 'build-tools', 'code_generator', 'functions.yaml')
    function_info = _read_yaml(functions_yaml_path)
    functions = parse_functions(function_info)
    cleanup_functions(functions)
    _write(os.path.join(root, 'source', 'nnabla-metadata.json'), json.dumps(functions, indent=2))

def _schema():
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    third_party_dir = os.path.join(root, 'third_party', 'source', 'nnabla')
    tmpl_file = os.path.join(third_party_dir, 'src/nbla/proto/nnabla.proto.tmpl')
    yaml_functions_path = os.path.join(third_party_dir, 'build-tools/code_generator/functions.yaml')
    yaml_solvers_path = os.path.join(third_party_dir, 'build-tools/code_generator/solvers.yaml')
    functions = _read_yaml(yaml_functions_path)
    function_info = {k: v for _, category in functions.items() for k, v in category.items()}
    solver_info = _read_yaml(yaml_solvers_path)
    _generate_from_template(tmpl_file, function_info=function_info, solver_info=solver_info)

def _try_eval_default(value):
    if value and isinstance(value, str) and not value.startswith(('(', '[')):
        if value is None or value == 'None':
            value = None
        elif value == 'True':
            value = True
        elif value == 'False':
            value = False
        elif value == 'list()':
            value = []
        elif len(value) > 2 and value[0] == "'" and value[-1] == "'":
            value = value[1:-1]
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
    return value

def main(): # pylint: disable=missing-function-docstring
    table = { 'metadata': _metadata, 'schema': _schema }
    for command in sys.argv[1:]:
        table[command]()

if __name__ == '__main__':
    main()
