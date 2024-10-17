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
    metadata = {}
    for value in json.loads(_read(metadata_file)):
        key = value['name']
        if key in metadata:
            raise ValueError(f"Duplicate key '{key}'")
        metadata[key] = value
    return metadata

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
    ('torch/csrc/jit/runtime/register_prim_ops.cpp',
        re.compile(r'(prim::.*->\s*.*)"', re.MULTILINE)),
    ('torch/csrc/jit/runtime/register_prim_ops_fulljit.cpp',
        re.compile(r'(aten::.*->\s*.*)"', re.MULTILINE)),
    ('torch/csrc/jit/runtime/register_special_ops.cpp',
        re.compile(r'(aten::.*->\s*.*)"', re.MULTILINE)),
    ('aten/src/ATen/native/RNN.cpp',
        re.compile(r'TORCH_SELECTIVE_SCHEMA\("(.*)"', re.MULTILINE)),
    ('torch/jit/_shape_functions.py',
        re.compile(r'(prim::.*->\s*.*)"', re.MULTILINE)),
    ('torch/csrc/jit/runtime/static/native_ops.cpp',
        re.compile(r'(prim::.*->\s*.*)"', re.MULTILINE)),
]

known_schema_definitions = [
    'aten::__and__.bool(bool a, bool b) -> bool',
    'aten::__and__.int(int a, int b) -> int',
    'aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::__getitem__.Dict_bool(Dict(bool, t) self, bool key) -> t(*)',
    'aten::__getitem__.Dict_complex(Dict(complex, t) self, complex key) -> t(*)',
    'aten::__getitem__.Dict_float(Dict(float, t) self, float key) -> t(*)',
    'aten::__getitem__.Dict_int(Dict(int, t) self, int key) -> t(*)',
    'aten::__getitem__.Dict_str(Dict(str, t) self, str key) -> t(*)',
    'aten::__getitem__.Dict_Tensor(Dict(Tensor, t) self, Tensor key) -> t(*)',
    'aten::__getitem__.str(str s, int index) -> str',
    'aten::__getitem__.t(t[](a) list, int idx) -> t(*)',
    'aten::any.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::any.bool(bool[] self) -> bool',
    'aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor',
    'aten::any.dimname_out(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)', # pylint: disable=line-too-long
    'aten::any.dimname(Tensor self, str dim, bool keepdim=False) -> Tensor',
    'aten::any.dims_out(Tensor self, int[]? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)', # pylint: disable=line-too-long
    'aten::any.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor',
    'aten::any.float(float[] self) -> bool',
    'aten::any.int(int[] self) -> bool',
    'aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::any.str(str[] self) -> bool',
    'aten::any(Tensor self) -> Tensor',
    'aten::as_tensor.bool(bool t, *, ScalarType? dtype=None, Device? device=None) -> Tensor',
    'aten::as_tensor.complex(complex t, *, ScalarType? dtype=None, Device? device=None) -> Tensor',
    'aten::as_tensor.float(float t, *, ScalarType? dtype=None, Device? device=None) -> Tensor',
    'aten::as_tensor.int(int t, *, ScalarType? dtype=None, Device? device=None) -> Tensor',
    'aten::as_tensor.list(t[] data, *, ScalarType? dtype=None, Device? device=None) -> Tensor',
    'aten::as_tensor(Tensor(a) data, *, ScalarType? dtype=None, Device? device=None) -> Tensor(b|a)', # pylint: disable=line-too-long
    'aten::ceil.float(float a) -> int',
    'aten::ceil.int(int a) -> int',
    'aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::ceil.Scalar(Scalar a) -> Scalar',
    'aten::ceil(Tensor self) -> Tensor',
    'aten::dict.bool((bool, tVal)[] inputs) -> Dict(bool, tVal)',
    'aten::dict.complex((complex, tVal)[] inputs) -> Dict(complex, tVal)',
    'aten::dict.Dict_bool(Dict(bool, t)(a) self) -> Dict(bool, t)',
    'aten::dict.Dict_complex(Dict(complex, t)(a) self) -> Dict(complex, t)',
    'aten::dict.Dict_float(Dict(float, t)(a) self) -> Dict(float, t)',
    'aten::dict.Dict_int(Dict(int, t)(a) self) -> Dict(int, t)',
    'aten::dict.Dict_str(Dict(str, t)(a) self) -> Dict(str, t)',
    'aten::dict.Dict_Tensor(Dict(Tensor, t)(a) self) -> Dict(Tensor, t)',
    'aten::dict.float((float, tVal)[] inputs) -> Dict(float, tVal)',
    'aten::dict.int((int, tVal)[] inputs) -> Dict(int, tVal)',
    'aten::dict.str((str, tVal)[] inputs) -> Dict(str, tVal)',
    'aten::dict.Tensor((Tensor, tVal)[] inputs) -> Dict(Tensor, tVal)',
    'aten::dict() -> Dict(str, Tensor)',
    'aten::div.complex(complex a, complex b) -> complex',
    'aten::div.float(float a, float b) -> float',
    'aten::div.int(int a, int b) -> float',
    'aten::div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)',  # pylint: disable=line-too-long
    'aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::div.Scalar_mode_out(Tensor self, Scalar other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)',  # pylint: disable=line-too-long
    'aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor',
    'aten::div.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::div.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor',
    'aten::div.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::div(Scalar a, Scalar b) -> float',
    'aten::eq_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::eq_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::eq.bool_list(bool[] a, bool[] b) -> bool',
    'aten::eq.bool(bool a, bool b) -> bool',
    'aten::eq.complex_float(complex a, float b) -> bool',
    'aten::eq.complex(complex a, complex b) -> bool',
    'aten::eq.device(Device a, Device b) -> bool',
    'aten::eq.enum(AnyEnumType a, AnyEnumType b) -> bool',
    'aten::eq.float_complex(float a, complex b) -> bool',
    'aten::eq.float_int(float a, int b) -> bool',
    'aten::eq.float_list(float[] a, float[] b) -> bool',
    'aten::eq.float(float a, float b) -> bool',
    'aten::eq.int_float(int a, float b) -> bool',
    'aten::eq.int_list(int[] a, int[] b) -> bool',
    'aten::eq.int(int a, int b) -> bool',
    'aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::eq.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::eq.str_list(str[] a, str[] b) -> bool',
    'aten::eq.str(str a, str b) -> bool',
    'aten::eq.Tensor_list(Tensor[] a, Tensor[] b) -> bool',
    'aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::eq.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::eq(Scalar a, Scalar b) -> bool',
    'aten::equal(Tensor self, Tensor other) -> bool',
    'aten::extend.t(t[](a!) self, t[] other) -> ()',
    'aten::gt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::gt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::gt.float_int(float a, int b) -> bool',
    'aten::gt.float(float a, float b) -> bool',
    'aten::gt.int_float(int a, float b) -> bool',
    'aten::gt.int(int a, int b) -> bool',
    'aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::gt.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::gt.str(str a, str b) -> bool',
    'aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::gt.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::gt(Scalar a, Scalar b) -> bool',
    'aten::item(Tensor self) -> Scalar',
    'aten::items.bool(Dict(bool, t) self) -> ((bool, t)[])',
    'aten::items.complex(Dict(complex, t) self) -> ((complex, t)[])',
    'aten::items.float(Dict(float, t) self) -> ((float, t)[])',
    'aten::items.int(Dict(int, t) self) -> ((int, t)[])',
    'aten::items.str(Dict(str, t) self) -> ((str, t)[])',
    'aten::items.Tensor(Dict(Tensor, t) self) -> ((Tensor, t)[])',
    'aten::keys.bool(Dict(bool, t) self) -> bool[](*)',
    'aten::keys.complex(Dict(complex, t) self) -> complex[](*)',
    'aten::keys.float(Dict(float, t) self) -> float[](*)',
    'aten::keys.int(Dict(int, t) self) -> int[](*)',
    'aten::keys.str(Dict(str, t) self) -> str[](*)',
    'aten::keys.Tensor(Dict(Tensor, t) self) -> Tensor[](*)',
    'aten::le_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::le_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::le.float_int(float a, int b) -> bool',
    'aten::le.float(float a, float b) -> bool',
    'aten::le.int_float(int a, float b) -> bool',
    'aten::le.int(int a, int b) -> bool',
    'aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::le.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::le.str(str a, str b) -> bool',
    'aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::le.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::le(Scalar a, Scalar b) -> bool',
    'aten::len.any(Any[] a) -> int',
    'aten::len.Dict_bool(Dict(bool, t) self) -> int',
    'aten::len.Dict_complex(Dict(complex, t) self) -> int',
    'aten::len.Dict_float(Dict(float, t) self) -> int',
    'aten::len.Dict_int(Dict(int, t) self) -> int',
    'aten::len.Dict_str(Dict(str, t) self) -> int',
    'aten::len.Dict_Tensor(Dict(Tensor, t) self) -> int',
    'aten::len.str(str s) -> int',
    'aten::len.t(t[] a) -> int',
    'aten::len.Tensor(Tensor t) -> int',
    'aten::log10.complex(complex a) -> complex',
    'aten::log10.float(float a) -> float',
    'aten::log10.int(int a) -> float',
    'aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::log10.Scalar(Scalar a) -> Scalar',
    'aten::log10(Tensor self) -> Tensor',
    'aten::lt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::lt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::lt.float_int(float a, int b) -> bool',
    'aten::lt.float(float a, float b) -> bool',
    'aten::lt.int_float(int a, float b) -> bool',
    'aten::lt.int(int a, int b) -> bool',
    'aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::lt.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::lt.str(str a, str b) -> bool',
    'aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::lt.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::lt(Scalar a, Scalar b) -> bool',
    'aten::pow.complex(complex a, complex b) -> complex',
    'aten::pow.complex_float(complex a, float b) -> complex',
    'aten::pow.float(float a, float b) -> float',
    'aten::pow.float_complex(float a, complex b) -> complex',
    'aten::pow.float_int(float a, int b) -> float',
    'aten::pow.int(int a, int b) -> float',
    'aten::pow.int_float(int a, float b) -> float',
    'aten::pow.int_to_int(int a, int b) -> int',
    'aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor',
    'aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::pow.Scalar_Scalar(Scalar a, Scalar b) -> float',
    'aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor',
    'aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor',
    'aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)',
    'aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)',
    'aten::remainder.float_int(float a, int b) -> float',
    'aten::remainder.float(float a, float b) -> float',
    'aten::remainder.int_float(int a, float b) -> float',
    'aten::remainder.int(int a, int b) -> int',
    'aten::remainder.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::remainder.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor',
    'aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::remainder.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::remainder(Scalar a, Scalar b) -> Scalar',
    'aten::replace(str self, str old, str new, int max=-1) -> str',
    'aten::searchsorted.Scalar_out(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!)',  # pylint: disable=line-too-long
    'aten::searchsorted.Scalar(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor',  # pylint: disable=line-too-long
    'aten::searchsorted.Tensor_out(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!)',  # pylint: disable=line-too-long
    'aten::searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor',  # pylint: disable=line-too-long
    'aten::sqrt.complex(complex a) -> complex',
    'aten::sqrt.float(float a) -> float',
    'aten::sqrt.int(int a) -> float',
    'aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::sqrt.Scalar(Scalar a) -> Scalar',
    'aten::sqrt(Tensor self) -> Tensor',
    'aten::values.bool(Dict(bool, t) self) -> t[](*)',
    'aten::values.complex(Dict(complex, t) self) -> t[](*)',
    'aten::values.float(Dict(float, t) self) -> t[](*)',
    'aten::values.int(Dict(int, t) self) -> t[](*)',
    'aten::values.str(Dict(str, t) self) -> t[](*)',
    'aten::values.Tensor(Dict(Tensor, t) self) -> t[](*)',
    'aten::values(Tensor(a) self) -> Tensor(a)',
    'prim::abs.complex(complex a) -> float',
    'prim::abs.float(float a) -> float',
    'prim::abs.int(int a) -> int',
    'prim::abs.Scalar(Scalar a) -> Scalar',
    'prim::abs(Tensor x) -> Tensor',
    'prim::device(Tensor a) -> Device',
    'prim::is_cpu(Tensor a) -> bool',
    'prim::is_cuda(Tensor a) -> bool',
    'prim::is_ipu(Tensor a) -> bool',
    'prim::is_maia(Tensor a) -> bool',
    'prim::is_meta(Tensor a) -> bool',
    'prim::is_mkldnn(Tensor a) -> bool',
    'prim::is_mps(Tensor a) -> bool',
    'prim::is_mtia(Tensor a) -> bool',
    'prim::is_nested(Tensor a) -> bool',
    'prim::is_quantized(Tensor a) -> bool',
    'prim::is_sparse_csr(Tensor a) -> bool',
    'prim::is_sparse(Tensor a) -> bool',
    'prim::is_vulkan(Tensor a) -> bool',
    'prim::is_xla(Tensor a) -> bool',
    'prim::is_xpu(Tensor a) -> bool',
    'prim::itemsize(Tensor a) -> int',
    'prim::layout(Tensor a) -> Layout',
    'prim::max.bool_list(bool[] l, bool[] r) -> bool[]',
    'prim::max.float_int(float a, int b) -> float',
    'prim::max.float_list(float[] l, float[] r) -> float[]',
    'prim::max.float(float a, float b) -> float',
    'prim::max.int_float(int a, float b) -> float',
    'prim::max.int_list(int[] l, int[] r) -> int[]',
    'prim::max.int(int a, int b) -> int',
    'prim::max.self_bool(bool[] self) -> bool',
    'prim::max.self_float(float[] self) -> float',
    'prim::max.self_int(int[] self) -> int',
    'prim::max(Scalar a, Scalar b) -> Scalar',
    'prim::min.bool_list(bool[] l, bool[] r) -> bool[]',
    'prim::min.float_int(float a, int b) -> float',
    'prim::min.float_list(float[] l, float[] r) -> float[]',
    'prim::min.float(float a, float b) -> float',
    'prim::min.int_float(int a, float b) -> float',
    'prim::min.int_list(int[] l, int[] r) -> int[]',
    'prim::min.int(int a, int b) -> int',
    'prim::min.self_bool(bool[] self) -> bool',
    'prim::min.self_float(float[] self) -> float',
    'prim::min.self_int(int[] self) -> int',
    'prim::min(Scalar a, Scalar b) -> Scalar',
]

def _parse_schemas():
    schemas = {}
    definitions = set()
    for entry in schema_source_files:
        path = os.path.join(pytorch_source_dir, entry[0])
        content = _read(path)
        content = content.splitlines()
        content = filter(lambda _: not _.startswith('#'), content)
        content = '\n'.join(content)
        for value in entry[1].findall(content):
            value = re.sub(r'\n|\r|\s*"', '', value) if value.startswith('_caffe2::') else value
            definition = entry[2] + value if len(entry) > 2 else value
            if not definition in definitions:
                definitions.add(definition)
                schema = pytorch.Schema(definition)
                if schema.name in schemas:
                    raise KeyError(schema.name)
                schemas[schema.name] = schema
    for definition in known_schema_definitions:
        schema = pytorch.Schema(definition)
        schemas[schema.name] = schema
    return schemas

def _filter_schemas(schemas, types):
    keys = set(map(lambda _: _.split('.')[0], types.keys()))
    filtered_schemas = set()
    for schema in schemas.values():
        for key in keys:
            if schema.name == key or schema.name.startswith(key + '.'):
                filtered_schemas.add(schema.name)
    for schema in schemas.values():
        if schema.name.startswith('aten::pop'):
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

def _check_types(types, schemas):
    types = dict(types.items())
    for schema in schemas.values():
        if schema.name in types:
            types.pop(schema.name)
    for key in list(types.keys()):
        if key.startswith('torch.nn') or key.startswith('__torch__.'):
            types.pop(key)
        if key.startswith('torchvision::') or \
           key.startswith('torchaudio::') or \
           key.startswith('neuron::'):
            types.pop(key)
        if key.startswith('_caffe2::'):
            types.pop(key)
    known_keys = [
        'aten::_native_batch_norm_legit_functional',
        'aten::add.float',
        'aten::add.int',
        'aten::add.str',
        'aten::arange.start_out_',
        'aten::classes._nnapi.Compilation',
        'aten::fft',
        'aten::floor.float',
        'aten::floor.int',
        'aten::floor.Scalar',
        'aten::floordiv.float_int',
        'aten::floordiv.float',
        'aten::floordiv.int_float',
        'aten::floordiv.int',
        'aten::floordiv.Scalar',
        'aten::grid_sampler.legacy',
        'aten::mul.float_int',
        'aten::mul.int_float',
        'aten::mul.int',
        'aten::mul.ScalarT',
        'aten::mul',
        'aten::ne.float',
        'aten::ne.int',
        'aten::ne.str',
        'aten::neg.complex',
        'aten::neg.float',
        'aten::neg.int',
        'aten::neg.Scalar',
        'aten::sub.float',
        'aten::sub.int',
        'aten::sub.str',
        'aten::tensor.bool',
        'aten::tensor.float',
        'aten::tensor.int',
        'prim::shape',
    ]
    for key in known_keys:
        types.pop(key)
    if len(types) > 0:
        raise Exception('\n'.join(list(types.keys()))) # pylint: disable=broad-exception-raised

def _metadata():
    types = _read_metadata()
    schemas = _parse_schemas()
    _check_types(types, schemas)
    _check_schemas(schemas)
    filtered_schemas = _filter_schemas(schemas, types)
    metadata = pytorch.Metadata(types)
    for schema in filtered_schemas.values():
        metadata.type(schema)
    _write_metadata(types)

def main(): # pylint: disable=missing-function-docstring
    _metadata()

if __name__ == '__main__':
    main()
