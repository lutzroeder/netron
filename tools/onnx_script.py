""" ONNX metadata script """

import collections
import json
import os
import re

import onnx.backend.test.case
import onnx.defs
import onnx.onnx_ml_pb2
import onnxruntime

attribute_type_table = [
    "undefined",
    "float32",
    "int64",
    "string",
    "tensor",
    "graph",
    "float32[]",
    "int64[]",
    "string[]",
    "tensor[]",
    "graph[]",
    "sparse_tensor",
    "sparse_tensor[]",
    "type_proto",
    "type_proto[]"
]


def _format_description(description):
    def replace_line(match):
        link = match.group(1)
        url = match.group(2)
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://github.com/onnx/onnx/blob/master/docs/" + url
        return "[" + link + "](" + url + ")"
    return re.sub('\\[(.+)\\]\\(([^ ]+?)( "(.+)")?\\)', replace_line, description)

def _format_range(value):
    return "&#8734;" if value == 2147483647 else str(value)

class OnnxSchema:
    """ ONNX schema """

    def __init__(self, schema, snippets):
        self.schema = schema
        self.snippets = snippets
        self.name = self.schema.name
        self.module = self.schema.domain if self.schema.domain else "ai.onnx"
        self.version = self.schema.since_version
        self.key = self.name + ":" + self.module + ":" + str(self.version).zfill(4)

    def _get_attr_type(self, attribute_type, attribute_name, op_type, op_domain):
        key = op_domain + ":" + op_type + ":" + attribute_name
        if key in (":Cast:to", ":EyeLike:dtype", ":RandomNormal:dtype"):
            return "DataType"
        return attribute_type_table[attribute_type]

    def _get_attr_default_value(self, attr_value):
        if attr_value.HasField("i"):
            return attr_value.i
        if attr_value.HasField("s"):
            return attr_value.s.decode("utf8")
        if attr_value.HasField("f"):
            return attr_value.f
        return None

    def _update_attributes(self, value, schema):
        target = value["attributes"] = []
        attributes = sorted(schema.attributes.items())
        for _ in collections.OrderedDict(attributes).values():
            value = {}
            value["name"] = _.name
            attr_type = self._get_attr_type(_.type, _.name, schema.name, schema.domain)
            if attr_type:
                value["type"] = attr_type
            value["required"] = _.required
            default_value = self._get_attr_default_value(_.default_value)
            if default_value:
                value["default"] = default_value
            description = _format_description(_.description)
            if len(description) > 0:
                value["description"] = description
            target.append(value)

    def _update_inputs(self, value, inputs):
        target = value["inputs"] = []
        for _ in inputs:
            value = {}
            value["name"] = _.name
            value["type"] = _.type_str
            if _.option == onnx.defs.OpSchema.FormalParameterOption.Optional:
                value["option"] = "optional"
            elif _.option == onnx.defs.OpSchema.FormalParameterOption.Variadic:
                value["list"] = True
            description = _format_description(_.description)
            if len(description) > 0:
                value["description"] = description
            target.append(value)

    def _update_outputs(self, value, outputs):
        target = value["outputs"] = []
        for _ in outputs:
            value = {}
            value["name"] = _.name
            value["type"] = _.type_str
            if _.option == onnx.defs.OpSchema.FormalParameterOption.Optional:
                value["option"] = "optional"
            elif _.option == onnx.defs.OpSchema.FormalParameterOption.Variadic:
                value["list"] = True
            description = _format_description(_.description)
            if len(description) > 0:
                value["description"] = description
            target.append(value)

    def _update_type_constraints(self, value, type_constraints):
        value["type_constraints"] = []
        for _ in type_constraints:
            value["type_constraints"].append({
                "description": _.description,
                "type_param_str": _.type_param_str,
                "allowed_type_strs": _.allowed_type_strs
            })

    def _update_snippets(self, value, snippets):
        target = value["examples"] = []
        for summary, code in sorted(snippets):
            lines = code.splitlines()
            while len(lines) > 0 and re.search("\\s*#", lines[-1]):
                lines.pop()
                if len(lines) > 0 and len(lines[-1]) == 0:
                    lines.pop()
            target.append({
                "summary": summary,
                "code": "\n".join(lines)
            })

    def to_dict(self):
        """ Serialize model to JSON message """
        value = {}
        value["name"] = self.name
        value["module"] = self.module
        value["version"] = self.version
        if self.schema.support_level != onnx.defs.OpSchema.SupportType.COMMON:
            value["status"] = self.schema.support_level.name.lower()
        description = _format_description(self.schema.doc.lstrip())
        if len(description) > 0:
            value["description"] = description
        if self.schema.attributes:
            self._update_attributes(value, self.schema)
        if self.schema.inputs:
            self._update_inputs(value, self.schema.inputs)
        value["min_input"] = self.schema.min_input
        value["max_input"] = self.schema.max_input
        if self.schema.outputs:
            self._update_outputs(value, self.schema.outputs)
        value["min_output"] = self.schema.min_output
        value["max_output"] = self.schema.max_output
        if self.schema.min_input != self.schema.max_input:
            value["inputs_range"] = _format_range(self.schema.min_input) + " - " \
                + _format_range(self.schema.max_input)
        if self.schema.min_output != self.schema.max_output:
            value["outputs_range"] = _format_range(self.schema.min_output) + " - " \
                + _format_range(self.schema.max_output)
        if self.schema.type_constraints:
            self._update_type_constraints(value, self.schema.type_constraints)
        if self.name in self.snippets:
            self._update_snippets(value, self.snippets[self.name])
        return value

class OnnxRuntimeSchema:
    """ ONNX Runtime schema """

    def __init__(self, schema):
        self.schema = schema
        self.name = self.schema.name
        self.module = self.schema.domain if self.schema.domain else "ai.onnx"
        self.version = self.schema.since_version
        self.key = self.name + ":" + self.module + ":" + str(self.version).zfill(4)

    def _get_attr_type(self, attribute_type):
        return attribute_type_table[attribute_type]

    def _get_attr_default_value(self, attr_value):
        if attr_value.HasField("i"):
            return attr_value.i
        if attr_value.HasField("s"):
            return attr_value.s.decode("utf8")
        if attr_value.HasField("f"):
            return attr_value.f
        return None

    def _update_attributes(self, value, schema):
        target = value["attributes"] = []
        attributes = sorted(schema.attributes.items())
        for _ in collections.OrderedDict(attributes).values():
            value = {}
            value["name"] = _.name
            attribute_type = self._get_attr_type(_.type)
            if attribute_type:
                value["type"] = attribute_type
            value["required"] = _.required
            default_value = onnx.onnx_ml_pb2.AttributeProto()
            default_value.ParseFromString(_._default_value)
            default_value = self._get_attr_default_value(default_value)
            if default_value:
                value["default"] = default_value
            description = _format_description(_.description)
            if len(description) > 0:
                value["description"] = description
            target.append(value)

    def _update_inputs(self, value, inputs):
        target = value["inputs"] = []
        for _ in inputs:
            value = {}
            value["name"] = _.name
            value["type"] = _.typeStr
            schemadef = onnxruntime.capi.onnxruntime_pybind11_state.schemadef
            if _.option == schemadef.OpSchema.FormalParameterOption.Optional:
                value["option"] = "optional"
            elif _.option == schemadef.OpSchema.FormalParameterOption.Variadic:
                value["list"] = True
            description = _format_description(_.description)
            if len(description) > 0:
                value["description"] = description
            target.append(value)

    def _update_outputs(self, value, outputs):
        target = value["outputs"] = []
        for _ in outputs:
            value = {}
            value["name"] = _.name
            value["type"] = _.typeStr
            schemadef = onnxruntime.capi.onnxruntime_pybind11_state.schemadef
            if _.option == schemadef.OpSchema.FormalParameterOption.Optional:
                value["option"] = "optional"
            elif _.option == schemadef.OpSchema.FormalParameterOption.Variadic:
                value["list"] = True
            description = _format_description(_.description)
            if len(description) > 0:
                value["description"] = description
            target.append(value)

    def _update_type_constraints(self, value, type_constraints):
        value["type_constraints"] = []
        for _ in type_constraints:
            value["type_constraints"].append({
                "description": _.description,
                "type_param_str": _.type_param_str,
                "allowed_type_strs": _.allowed_type_strs
            })

    def to_dict(self):
        """ Serialize model to JSON message """
        value = {}
        value["name"] = self.name
        value["module"] = self.module
        value["version"] = self.version
        schemadef = onnxruntime.capi.onnxruntime_pybind11_state.schemadef
        if self.schema.support_level != schemadef.OpSchema.SupportType.COMMON:
            value["status"] = self.schema.support_level.name.lower()
        if self.schema.doc:
            description = _format_description(self.schema.doc.lstrip())
            if len(description) > 0:
                value["description"] = description
        if self.schema.attributes:
            self._update_attributes(value, self.schema)
        if self.schema.inputs:
            self._update_inputs(value, self.schema.inputs)
        value["min_input"] = self.schema.min_input
        value["max_input"] = self.schema.max_input
        if self.schema.outputs:
            self._update_outputs(value, self.schema.outputs)
        value["min_output"] = self.schema.min_output
        value["max_output"] = self.schema.max_output
        if self.schema.min_input != self.schema.max_input:
            value["inputs_range"] = _format_range(self.schema.min_input) + " - " \
                + _format_range(self.schema.max_input)
        if self.schema.min_output != self.schema.max_output:
            value["outputs_range"] = _format_range(self.schema.min_output) + " - " \
                + _format_range(self.schema.max_output)
        if self.schema.type_constraints:
            self._update_type_constraints(value, self.schema.type_constraints)
        return value

def _metadata():
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    file = os.path.join(root_dir, "source", "onnx-metadata.json")
    with open(file, encoding="utf-8") as handle:
        content = handle.read()
    categories = {}
    content = json.loads(content)
    for schema in content:
        if "category" in schema:
            name = schema["name"]
            categories[name] = schema["category"]
    types = collections.OrderedDict()
    numpy = __import__("numpy")
    with numpy.errstate(all="ignore"):
        snippets = onnx.backend.test.case.collect_snippets()
    for schema in onnx.defs.get_all_schemas_with_history():
        schema = OnnxSchema(schema, snippets)
        if schema.key not in types:
            types[schema.key] = schema.to_dict()
    for schema in onnxruntime.capi.onnxruntime_pybind11_state.get_all_operator_schema():
        schema = OnnxRuntimeSchema(schema)
        if schema.key not in types:
            types[schema.key] = schema.to_dict()
    for schema in content:
        key = f"{schema['name']}:{schema['module']}:{str(schema['version']).zfill(4)}"
        if key not in types:
            types[key] = schema
    types = [types[key] for key in sorted(types)]
    for schema in types:
        name = schema["name"]
        # copy = schema.copy()
        # schema.clear()
        # schema['name'] = name
        # schema['module'] = copy['module']
        if name in categories:
            schema["category"] = categories[name]
        # for key, value in copy.items():
        #     if key not in schema:
        #         schema[key] = value
    content = json.dumps(types, indent=2)
    with open(file, "w", encoding="utf-8") as handle:
        handle.write(content)

def main():
    _metadata()

if __name__ == "__main__":
    main()
