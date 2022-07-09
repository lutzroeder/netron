''' NNabla metadata script '''

import json
import sys
import os
import yaml
import mako
import mako.template

def _render_with_template(text=None, filename=None, preprocessor=None, template_kwargs={}):
    tmpl = mako.template.Template(text=text, filename=filename, preprocessor=preprocessor)
    return tmpl.render(**template_kwargs)

def _generate_from_template(path_template, **kwargs):
    path_out = path_template.replace('.tmpl', '')
    generated = _render_with_template(filename=path_template, template_kwargs=kwargs)
    with open(path_out, 'wb') as file:
        write_content = generated.encode('utf_8')
        write_content = write_content.replace(b'\r\n', b'\n')
        write_content = write_content.replace(b'\r', b'\n')
        file.write(write_content)

def _metadata():
    json_file = os.path.join(os.path.dirname(__file__), '../source/nnabla-metadata.json')
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../third_party/source/nnabla'))
    yaml_functions = os.path.join(base, 'build-tools/code_generator/functions.yaml')
    with open(yaml_functions, 'r', encoding='utf-8') as file:
        function_info = yaml.safe_load(file)
    functions = []
    # parse functions
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
    # clean-up redundant fields
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
    with open(json_file, 'w', encoding='utf-8') as file:
        file.write(json.dumps(functions, indent=2))

def _schema():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../third_party/source/nnabla'))
    tmpl_file = os.path.join(base, 'src/nbla/proto/nnabla.proto.tmpl')
    yaml_functions = os.path.join(base, 'build-tools/code_generator/functions.yaml')
    yaml_solvers = os.path.join(base, 'build-tools/code_generator/solvers.yaml')
    with open(yaml_functions, 'r', encoding='utf-8') as file:
        functions = yaml.safe_load(file)
        function_info = {k: v for _, category in functions.items() for k, v in category.items()}
    with open(yaml_solvers, 'r', encoding='utf-8') as file:
        solver_info = yaml.safe_load(file)
    _generate_from_template(tmpl_file, function_info=function_info, solver_info=solver_info)

def _try_eval_default(default):
    if default and isinstance(default, str):
        if not default.startswith(('(', '[')):
            try:
                default = eval(default, {'__builtin__': None})
            except NameError:
                pass
    return default

def main(): # pylint: disable=missing-function-docstring
    command_table = {'metadata': _metadata, 'schema': _schema}
    command = sys.argv[1]
    command_table[command]()

if __name__ == '__main__':
    main()
