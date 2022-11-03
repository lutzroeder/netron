''' TorchScript metadata script '''

import collections
import json
import os
import re
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.pycache_prefix = os.path.join(root_dir, 'dist', 'pycache', 'test', 'backend')
pytorch = __import__('source.pytorch').pytorch

source_dir = os.path.join(root_dir, 'source')
third_party_dir = os.path.join(root_dir, 'third_party')
metadata_file = os.path.join(source_dir, 'pytorch-metadata.json')
pytorch_source_dir = os.path.join(third_party_dir, 'source', 'pytorch')

def _read(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

def _write(path, content):
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)

def _read_metadata():
    metadata: list[dict[str,object]] = json.loads(_read(metadata_file))
    return dict(map(lambda _: ( _['name'], _ ), metadata))

def _write_metadata(value):
    metadata = list(collections.OrderedDict(sorted(value.items())).values())
    content = json.dumps(metadata, indent=2, ensure_ascii=False)
    content = re.sub(r'\s {8}', ' ', content)
    content = re.sub(r',\s {8}', ', ', content)
    content = re.sub(r'\s {6}}', ' }', content)
    _write(metadata_file, content)

schema_source_files = [
    ('aten/src/ATen/native/native_functions.yaml',
        re.compile(r'-\s*func:\s*(.*)', re.MULTILINE), 'aten::'),
    ('aten/src/ATen/native/quantized/library.cpp',
        re.compile(r'TORCH_SELECTIVE_SCHEMA\("(.*)"\)', re.MULTILINE)),
    ('aten/src/ATen/native/xnnpack/RegisterOpContextClass.cpp',
        re.compile(r'TORCH_SELECTIVE_SCHEMA\("(.*)"', re.MULTILINE)),
    ('torch/csrc/jit/runtime/register_prim_ops.cpp',
        re.compile(r'(aten::.*->\s*.*)"', re.MULTILINE)),
    ('torch/csrc/jit/runtime/register_prim_ops_fulljit.cpp',
        re.compile(r'(aten::.*->\s*.*)"', re.MULTILINE)),
    ('torch/csrc/jit/runtime/register_special_ops.cpp',
        re.compile(r'(aten::.*->\s*.*)"', re.MULTILINE)),
    ('caffe2/operators/copy_op.cc',
        re.compile(r'(_caffe2::.*->\s*Tensor)', re.MULTILINE)),
    ('caffe2/operators/batch_permutation_op.cc',
        re.compile(r'(_caffe2::.*->\s*Tensor)', re.MULTILINE)),
    ('caffe2/operators/collect_and_distribute_fpn_rpn_proposals_op.cc',
        re.compile(r'"(_caffe2::[\w+]*\([\w"\s\[\],]*\)\s*->\s*\([\w"\s\[\],]*\))"', re.MULTILINE)),
    ('caffe2/operators/box_with_nms_limit_op.cc',
        re.compile(r'"(_caffe2::[\w+]*\([\w"\s\[\],]*\)\s*->\s*\([\w"\s\[\],]*\))"', re.MULTILINE)),
    ('caffe2/operators/bbox_transform_op.cc',
        re.compile(r'"(_caffe2::[\w+]*\([\w"\s\[\],]*\)\s*->\s*\([\w"\s\[\],]*\))"', re.MULTILINE)),
    ('caffe2/operators/generate_proposals_op.cc',
        re.compile(r'"(_caffe2::[\w+]*\([\w"\s\[\],]*\)\s*->\s*\([\w"\s\[\],]*\))"', re.MULTILINE)),
    ('caffe2/operators/roi_align_op.cc',
        re.compile(r'"(_caffe2::[\w+]*\([\w"\s\[\],]*\)\s*->.*)"', re.MULTILINE))
]

def _parse_schemas():
    schemas = {}
    for entry in schema_source_files:
        path = os.path.join(pytorch_source_dir, entry[0])
        content = _read(path)
        for value in entry[1].findall(content):
            value = re.sub(r'\n|\r|\s*"', '', value) if value.startswith('_caffe2::') else value
            definition = entry[2] + value if len(entry) > 2 else value
            schema = pytorch.Schema(definition)
            if schema.name in schemas:
                raise Exception()
            schemas[schema.name] = schema
    return schemas

def _filter_schemas(schemas, types):

    keys = set(map(lambda _: _.split('.')[0], types.keys()))
    filtered_schemas = set()
    for schema in schemas.values():
        for key in keys:
            if schema.name == key or schema.name.startswith(key + '.'):
                filtered_schemas.add(schema.name)
    # filtered_schemas = set(types.keys())
    # content = _read('list.csv')
    # regex = re.compile(r'Unsupported function \'(.*)\' in', re.MULTILINE)
    # matches = set()
    # for match in regex.findall(content):
    #     if match.startswith('torch.'):
    #         matches.add('aten::' + match[6:])
    #     if match.startswith('ops.') and len(match.split('.')) > 2:
    #         matches.add(match[4:].replace('.', '::'))
    # for schema in schemas.values():
    #     for match in matches:
    #         if schema.name.startswith(match):
    #             filtered_schemas.add(schema.name)
    return dict(filter(lambda _: _[0] in filtered_schemas, schemas.items()))

def _check_schemas(schemas): # pylint: disable=unused-argument
    # import torch
    # for name in dir(torch.ops.aten):
    #     if name.startswith('__') or name == 'name':
    #         continue
    #     packet = getattr(torch.ops.aten, name)
    #     for overload in packet.overloads():
    #         key = 'aten::' + name + ('.' + overload if overload != 'default' else '')
    #         overload_schema = str(getattr(packet, overload)._schema)
    #         if key in schemas:
    #             schema = schemas[key]
    #             if overload_schema != str(schema):
    #                 print(overload_schema)
    #                 print(schema)
    pass

def _translate_type(value):
    optional = False
    argument_type = ''
    while not isinstance(value, str):
        if isinstance(value, pytorch.Schema.OptionalType):
            value = value.element_type
            optional = True
        elif isinstance(value, pytorch.Schema.ListType):
            size = str(value.size) if hasattr(value, 'size') else ''
            value = value.element_type
            argument_type = '[' + size + ']' + argument_type
        else:
            name = value.name
            if name == 'int':
                name = 'int64'
            elif name == 'float':
                name = 'float32'
            elif name == 'bool':
                name = 'boolean'
            elif name == 'str':
                name = 'string'
            argument_type = name + argument_type
            break
    return argument_type, optional

def _check_types(types, schemas):
    types = dict(types.items())
    for schema in schemas.values():
        if schema.name in types:
            types.pop(schema.name)
    for key in list(types.keys()):
        if key.startswith('torch.nn'):
            types.pop(key)
        if key.startswith('torchvision::') or \
           key.startswith('torchaudio::') or \
           key.startswith('neuron::'):
            types.pop(key)
    types.pop('aten::fft')
    types.pop('aten::mul.ScalarT')
    types.pop('aten::classes._nnapi.Compilation')
    if len(types) > 0:
        print('\n'.join(list(types.keys())))

def _metadata():
    types = _read_metadata()
    schemas = _parse_schemas()
    _check_types(types, schemas)
    _check_schemas(schemas)
    filtered_schemas = _filter_schemas(schemas, types)
    for schema in filtered_schemas.values():
        arguments = list(filter(lambda argument: not argument.is_out, schema.arguments))
        returns = schema.returns
        if schema.name not in types:
            types[schema.name] = {
                'name': schema.name,
                'inputs': [ {} for i in range(len(arguments)) ],
                'outputs': [ {} for i in range(len(returns)) ],
            }
        value = types[schema.name]
        if schema.name == 'aten::as_tensor':
            continue
        if len(arguments) != len(value['inputs']) or len(returns) != len(value['outputs']):
            raise Exception(schema.name)
        for i, _ in enumerate(arguments):
            argument = value['inputs'][i]
            argument['name'] = _.name
            argument_type, optional = _translate_type(getattr(_, 'type'))
            argument['type'] = argument_type
            if optional:
                argument['optional'] = True
            if hasattr(_, 'default'):
                argument['default'] = _.default
        for i, _ in enumerate(returns):
            argument = value['outputs'][i]
            if hasattr(_, 'name'):
                value['outputs'][i]['name'] = _.name
            argument_type, optional = _translate_type(getattr(_, 'type'))
            argument['type'] = argument_type
            if optional:
                argument['optional'] = True
    _write_metadata(types)

def main(): # pylint: disable=missing-function-docstring
    _metadata()

if __name__ == '__main__':
    main()
