import json
import os
import sys
import yaml


def metadata():
    json_file = os.path.join(os.path.dirname(__file__), './source/nnabla-metadata.json')
    yaml_functions = os.path.join(os.path.dirname(__file__), './third_party/source/nnabla/build-tools/code_generator/functions.yaml')
    yaml_solvers = os.path.join(os.path.dirname(__file__), './third_party/source/nnabla/build-tools/code_generator/solvers.yaml')

    with open(yaml_functions, 'r') as file:
        functions = yaml.safe_load(file)

    with open(yaml_solvers, 'r') as file:
        solvers = yaml.safe_load(file)

    nnabla_metadata = {
        'functions': [],
        'solvers': []
    }

    # Parse functions
    for category_name, category in functions.items():
        for function_name, function_value in category.items():
            function = {
                'function_name': function_name,
                'name': function_value['snake_name'],
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
                    'default': try_eval_default(arg_value.get('default', None)),
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
            nnabla_metadata['functions'].append(function)

    # Clean-up redundant fields
    for function in nnabla_metadata['functions']:
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

    # Parse solvers
    for solver_name, solver_value in solvers.items():
        solver = {
            'solver_name': solver_name,
            'name': solver_value['snake_name'],
            'description': solver_value['doc'].strip()
        }
        for arg_name, arg_value in solver_value.get('arguments', {}).items():
            solver.setdefault('attributes', []).append({
                'name': arg_name,
                'type': arg_value['type'],
                'default': try_eval_default(arg_value.get('default', None)),
                'description': arg_value['doc'].strip()
            })
        nnabla_metadata['solvers'].append(solver)

    js_proto = os.path.join(os.path.dirname(__file__), './source/nnabla-proto.js')

    # Function case statements
    function_cases = []
    for function in nnabla_metadata['functions']:
        autogen_case_statement(function['function_name'], function['name'], function_cases)

    # Solver case statements
    solver_cases = []
    for solver in nnabla_metadata['solvers']:
        autogen_case_statement(solver['solver_name'], solver['name'], solver_cases)

    # Generate proto function/solver parameters (overwriting existing when previously generated)
    with open(js_proto, 'r') as file:
        lines = file.readlines()
        parameter_prefix = '$root.nnabla.AffineParameter = class AffineParameter'
        start_function_parameters = -1
        start_function_cases = -1
        end_function_cases = -1
        start_solver_cases = -1
        end_solver_cases = -1
        for index in range(len(lines)):
            if lines[index].startswith(parameter_prefix):
                start_function_parameters = index
            if '// Function parameter messages - Start' in lines[index]:
                start_function_cases = index + 1
            if '// Function parameter messages - End' in lines[index]:
                end_function_cases = index
            if '// Solver parameter messages - Start' in lines[index]:
                start_solver_cases = index + 1
            if '// Solver parameter messages - End' in lines[index]:
                end_solver_cases = index

    with open(js_proto, 'w') as file:
        file.writelines(lines[:start_function_cases])
        file.writelines(function_cases)
        file.writelines(lines[end_function_cases:start_solver_cases])
        file.writelines(solver_cases)
        file.writelines(lines[end_solver_cases:start_function_parameters])
        for function in nnabla_metadata['functions']:
            autogen_parameter_class(file, function['function_name'], function.get('attributes', []))
        for solver in nnabla_metadata['solvers']:
            autogen_parameter_class(file, solver['solver_name'], solver.get('attributes', []))

    # Remove function_name attributes and replace name attributes
    for function in nnabla_metadata['functions']:
        function['name'] = function.pop('function_name', None)

    with open(json_file, 'w') as file:
        file.write(json.dumps(nnabla_metadata['functions'], indent=2))


def autogen_case_statement(name: str, snake_name: str, dest):
    dest.append(f'                case "{snake_name}_param":\n')
    dest.append(f'                    message.{snake_name}_param = $root.nnabla.{name}Parameter.decodeText(reader);\n')
    dest.append(f'                    break;\n')


def autogen_parameter_class(file, name: str, attributes):
    file.write(f'$root.nnabla.{name}Parameter = class {name}Parameter {{\n\n')
    file.write(f'    constructor() {{\n')

    any_initialized = False
    for attribute in attributes:
        attribute_name = attribute['name']
        attribute_type = attribute['type']
        if attribute_type.startswith('repeated'):
            file.write(f'        this.{attribute_name} = [];\n')
            any_initialized = True
    if not any_initialized:
        file.write(f'\n')

    file.write(f'    }}\n\n')
    file.write(f'    static decodeText(reader) {{\n')
    file.write(f'        const message = new $root.nnabla.{name}Parameter();\n')
    file.write(f'        reader.start();\n')
    file.write(f'        while (!reader.end()) {{\n')
    file.write(f'            const tag = reader.tag();\n')
    file.write(f'            switch (tag) {{\n')

    for attribute in attributes:
        attribute_name = attribute['name']
        attribute_type = attribute['type']
        file.write(f'                case "{attribute_name}":\n')
        file.write(f'                    message.{attribute_name}')
        if attribute_type in ('bool', 'double', 'float', 'int64', 'string'):
            file.write(f' = reader.{attribute_type}();\n')
        elif attribute_type == 'Communicator':
            file.write(f' = $root.nnabla.Communicator.decodeText(reader);\n')
        elif attribute_type == 'Shape':
            file.write(f' = $root.nnabla.Shape.decodeText(reader);\n')
        elif attribute_type.startswith('repeated'):
            file.write(f'.push(reader.{attribute_type.split(" ")[1]}());\n')
        else:
            assert False, f'Failed to find the attribute type: {attribute_type}'
        file.write(f'                    break;\n')

    file.write(f'                default:\n')
    file.write(f'                    reader.field(tag, message);\n')
    file.write(f'                    break;\n')
    file.write(f'            }}\n')
    file.write(f'        }}\n')
    file.write(f'        return message;\n')
    file.write(f'    }}\n')
    file.write(f'}};\n\n')

    any_initialized = False
    for attribute in attributes:
        attribute_name = attribute['name']
        attribute_type = attribute['type']
        if not attribute_type.startswith('repeated'):
            if attribute_type == 'Shape':
                default = 'null'
            else:
                attribute_default = attribute.get('default')
                if attribute_type == 'bool':
                    default = 'true' if isinstance(attribute_default, bool) and attribute_default else 'false'
                elif attribute_type in ('double', 'float'):
                    default = attribute_default if isinstance(attribute_default, float) else '0.0'
                elif attribute_type == 'int64':
                    default = attribute_default if isinstance(attribute_default, int) else '0'
                elif attribute_type == 'string':
                    default = f'"{attribute_default}"' if isinstance(attribute_default, str) else '""'
                else:
                    continue
            file.write(f'$root.nnabla.{name}Parameter.prototype.{attribute_name} = {default};\n')
            any_initialized = True
    if any_initialized:
        file.write(f'\n')


def try_eval_default(default):
    if default and isinstance(default, str):
        if not default.startswith(('(', '[')):
            try:
                return eval(default, {'__builtin__': None})
            except NameError:
                pass
    return default


if __name__ == '__main__':
    command_table = { 'metadata': metadata }
    command = sys.argv[1]
    command_table[command]()
