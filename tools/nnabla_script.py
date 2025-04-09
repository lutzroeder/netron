""" NNabla metadata script """

import json
import os
import sys

import mako.template
import yaml


def _write(path, content):
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)

def _read_yaml(path):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)

def _metadata():
    def parse_functions(function_info):
        functions = []
        for category_name, category in function_info.items():
            for function_name, function_value in category.items():
                function = {
                    "name": function_name,
                    "description": function_value["doc"].strip()
                }
                for input_name, input_value in function_value.get("inputs", {}).items():
                    option = "optional" if input_value.get("optional", False) else None
                    variadic = input_value.get("variadic", False)
                    function.setdefault("inputs", []).append({
                        "name": input_name,
                        "type": "nnabla.Variable",
                        "option": option,
                        "list": variadic,
                        "description": input_value["doc"].strip()
                    })
                for arg_name, arg_value in function_value.get("arguments", {}).items():
                    attribute = _attribute(arg_name, arg_value)
                    function.setdefault("attributes", []).append(attribute)
                outputs = function_value.get("outputs", {})
                for output_name, output_value in outputs.items():
                    function.setdefault("outputs", []).append({
                        "name": output_name,
                        "type": "nnabla.Variable",
                        "list": output_value.get("variadic", False),
                        "description": output_value["doc"].strip()
                    })
                if "Pooling" in function_name:
                    function["category"] = "Pool"
                elif category_name == "Neural Network Layer":
                    function["category"] = "Layer"
                elif category_name == "Neural Network Activation Functions":
                    function["category"] = "Activation"
                elif category_name == "Normalization":
                    function["category"] = "Normalization"
                elif category_name == "Logical":
                    function["category"] = "Logic"
                elif category_name == "Array Manipulation":
                    function["category"] = "Shape"
                functions.append(function)
        return functions
    def cleanup_functions(functions):
        for function in functions:
            for inp in function.get("inputs", []):
                if inp["option"] is None:
                    inp.pop("option", None)
                if not inp["list"]:
                    inp.pop("list", None)
            for output in function.get("outputs", []):
                if not output["list"]:
                    output.pop("list", None)
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    nnabla_dir = os.path.join(root, "third_party", "source", "nnabla")
    code_generator_dir = os.path.join(nnabla_dir, "build-tools", "code_generator")
    functions_yaml_path = os.path.join(code_generator_dir, "functions.yaml")
    function_info = _read_yaml(functions_yaml_path)
    functions = parse_functions(function_info)
    cleanup_functions(functions)
    metadata_file = os.path.join(root, "source", "nnabla-metadata.json")
    metadata = json.dumps(functions, indent=2)
    _write(metadata_file, metadata)

def _schema():
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    nnabla_dir = os.path.join(root, "third_party", "source", "nnabla")
    tmpl_file = os.path.join(nnabla_dir, "src/nbla/proto/nnabla.proto.tmpl")
    code_generator_dir = os.path.join(nnabla_dir, "build-tools", "code_generator")
    yaml_functions_path = os.path.join(code_generator_dir, "functions.yaml")
    yaml_solvers_path = os.path.join(code_generator_dir, "solvers.yaml")
    functions = _read_yaml(yaml_functions_path)
    function_info = {}
    for _, category in functions.items():
        function_info.update(category)
    solver_info = _read_yaml(yaml_solvers_path)
    path = tmpl_file.replace(".tmpl", "")
    template = mako.template.Template(text=None, filename=tmpl_file, preprocessor=None)
    content = template.render(function_info=function_info, solver_info=solver_info)
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    _write(path, content)

def _attribute(name, value):
    attribute = {}
    attribute["name"] = name
    default = "default" in value
    if not default:
        attribute["required"] = True
    if value["type"] == "float":
        attribute["type"] = "float32"
        if default:
            attribute["default"] = float(value["default"])
    elif value["type"] == "double":
        attribute["type"] = "float64"
        if default:
            attribute["default"] = float(value["default"])
    elif value["type"] == "bool":
        attribute["type"] = "boolean"
        if default:
            _ = value["default"]
            if isinstance(_, bool):
                attribute["default"] = _
            elif _ == "True":
                attribute["default"] = True
            elif _ == "False":
                attribute["default"] = False
    elif value["type"] == "string":
        attribute["type"] = "string"
        if default:
            _ = value["default"]
            attribute["default"] = _.strip("'")
    elif value["type"] == "int64":
        attribute["type"] = "int64"
        if default:
            _ = value["default"]
            if isinstance(_, str) and not _.startswith("len") and _ != "None":
                attribute["default"] = int(_)
            else:
                attribute["default"] = _
    elif value["type"] == "repeated int64":
        attribute["type"] = "int64[]"
    elif value["type"] == "repeated float":
        attribute["type"] = "float32[]"
    elif value["type"] == "Shape":
        attribute["type"] = "shape"
    if default and "default" not in attribute:
        attribute["default"] = value["default"]
    attribute["description"] = value["doc"].strip()
    return attribute

def main():
    table = { "metadata": _metadata, "schema": _schema }
    for command in sys.argv[1:]:
        table[command]()

if __name__ == "__main__":
    main()
