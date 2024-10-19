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
        key = key.split("(")[0]
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

# pylint: disable=line-too-long
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
    'aten::_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)',
    'aten::_native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)',
    'aten::_native_batch_norm_legit.no_stats_out(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))',
    'aten::_native_batch_norm_legit.out(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps, *, Tensor(d!) out, Tensor(e!) save_mean, Tensor(f!) save_invstd) -> (Tensor(d!), Tensor(e!), Tensor(f!))',
    'aten::_native_batch_norm_legit_functional(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor running_mean_out, Tensor running_var_out)',
    'aten::_native_batch_norm_legit_no_training(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)',
    'aten::_native_batch_norm_legit_no_training.out(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))',
    'aten::_native_multi_head_attention(Tensor query, Tensor key, Tensor value, int embed_dim, int num_head, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, Tensor? mask=None, bool need_weights=True, bool average_attn_weights=True, int? mask_type=None) -> (Tensor, Tensor)',
    'aten::_native_multi_head_attention.out(Tensor query, Tensor key, Tensor value, int embed_dim, int num_head, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, Tensor? mask=None, bool need_weights=True, bool average_attn_weights=True, int? mask_type=None, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))',
    'aten::add(Scalar a, Scalar b) -> Scalar',
    'aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor',
    'aten::add.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor',
    'aten::add.complex(complex a, complex b) -> complex',
    'aten::add.complex_float(complex a, float b) -> complex',
    'aten::add.complex_int(complex a, int b) -> complex',
    'aten::add.float(float a, float b) -> float',
    'aten::add.float_complex(float a, complex b) -> complex',
    'aten::add.float_int(float a, int b) -> float',
    'aten::add.int(int a, int b) -> int',
    'aten::add.int_complex(int a, complex b) -> complex',
    'aten::add.int_float(int a, float b) -> float',
    'aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)',
    'aten::add.str(str a, str b) -> str',
    'aten::add.t(t[] a, t[] b) -> t[]',
    'aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)',
    'aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)',
    'aten::add_.t(t[](a!) self, t[] b) -> t[]',
    'aten::any.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::any.bool(bool[] self) -> bool',
    'aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor',
    'aten::any.dimname_out(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::any.dimname(Tensor self, str dim, bool keepdim=False) -> Tensor',
    'aten::any.dims_out(Tensor self, int[]? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)',
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
    'aten::as_tensor(Tensor(a) data, *, ScalarType? dtype=None, Device? device=None) -> Tensor(b|a)',
    'aten::bitwise_and.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_and.Scalar_Tensor(Scalar self, Tensor other) -> Tensor',
    'aten::bitwise_and.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_and_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::bitwise_and_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::bitwise_left_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor',
    'aten::bitwise_left_shift.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::bitwise_left_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::bitwise_left_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_left_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::bitwise_left_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::bitwise_not(Tensor self) -> Tensor',
    'aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)',
    'aten::bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::bitwise_or.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_or.Scalar_Tensor(Scalar self, Tensor other) -> Tensor',
    'aten::bitwise_or.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::bitwise_or.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_or_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::bitwise_or_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::bitwise_right_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor',
    'aten::bitwise_right_shift.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::bitwise_right_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::bitwise_right_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_right_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::bitwise_right_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::bitwise_xor.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_xor.Scalar_Tensor(Scalar self, Tensor other) -> Tensor',
    'aten::bitwise_xor.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::bitwise_xor.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::bitwise_xor_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::bitwise_xor_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::Bool.float(float a) -> bool',
    'aten::Bool.int(int a) -> bool',
    'aten::Bool.Tensor(Tensor a) -> bool',
    'aten::ceil.float(float a) -> int',
    'aten::ceil.int(int a) -> int',
    'aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::ceil.Scalar(Scalar a) -> Scalar',
    'aten::ceil(Tensor self) -> Tensor',
    'aten::complex(Tensor real, Tensor imag) -> Tensor',
    'aten::Complex.bool_bool(bool x, bool y) -> complex',
    'aten::Complex.bool_float(bool x, float y) -> complex',
    'aten::Complex.bool_int(bool x, int y) -> complex',
    'aten::Complex.bool_Tensor(bool x, Tensor y) -> complex',
    'aten::Complex.float_bool(float x, bool y) -> complex',
    'aten::Complex.float_float(float x, float y) -> complex',
    'aten::Complex.float_int(float x, int y) -> complex',
    'aten::Complex.float_Tensor(float x, Tensor y) -> complex',
    'aten::Complex.int_bool(int x, bool y) -> complex',
    'aten::Complex.int_float(int x, float y) -> complex',
    'aten::Complex.int_int(int x, int y) -> complex',
    'aten::Complex.int_Tensor(int x, Tensor y) -> complex',
    'aten::complex.out(Tensor real, Tensor imag, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::Complex.Scalar(Scalar a) -> complex',
    'aten::Complex.Tensor_bool(Tensor x, bool y) -> complex',
    'aten::Complex.Tensor_float(Tensor x, float y) -> complex',
    'aten::Complex.Tensor_int(Tensor x, int y) -> complex',
    'aten::Complex.Tensor_Tensor(Tensor a, Tensor b) -> complex',
    'aten::ComplexImplicit(Tensor a) -> complex',
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
    'aten::div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)',
    'aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::div.Scalar_mode_out(Tensor self, Scalar other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)',
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
    'aten::Float.bool(bool a) -> float',
    'aten::Float.int(int a) -> float',
    'aten::Float.Scalar(Scalar a) -> float',
    'aten::Float.str(str a) -> float',
    'aten::Float.Tensor(Tensor a) -> float',
    'aten::floor(Tensor self) -> Tensor',
    'aten::floor.Scalar(Scalar a) -> Scalar',
    'aten::floor.float(float a) -> int',
    'aten::floor.int(int a) -> int',
    'aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::floor_(Tensor(a!) self) -> Tensor(a!)',
    'aten::floor_divide(Tensor self, Tensor other) -> Tensor',
    'aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::floor_divide.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::floor_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::floordiv(Scalar a, Scalar b) -> Scalar',
    'aten::floordiv.float(float a, float b) -> float',
    'aten::floordiv.float_int(float a, int b) -> float',
    'aten::floordiv.int(int a, int b) -> int',
    'aten::floordiv.int_float(int a, float b) -> float',
    'aten::fmax(Tensor self, Tensor other) -> Tensor',
    'aten::fmax.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::fmin(Tensor self, Tensor other) -> Tensor',
    'aten::fmin.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::fmod(Scalar a, Scalar b) -> float',
    'aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::fmod.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::fmod.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::fmod.float(float a, float b) -> float',
    'aten::fmod.float_int(float a, int b) -> float',
    'aten::fmod.int(int a, int b) -> float',
    'aten::fmod.int_float(int a, float b) -> float',
    'aten::fmod_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::fmod_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::get.bool(Dict(bool, t) self, bool key) -> t(*)?',
    'aten::get.complex(Dict(complex, t) self, complex key) -> t(*)?',
    'aten::get.default_bool(Dict(bool, t) self, bool key, t default_value) -> t(*)',
    'aten::get.default_complex(Dict(complex, t) self, complex key, t default_value) -> t(*)',
    'aten::get.default_float(Dict(float, t) self, float key, t default_value) -> t(*)',
    'aten::get.default_int(Dict(int, t) self, int key, t default_value) -> t(*)',
    'aten::get.default_str(Dict(str, t) self, str key, t default_value) -> t(*)',
    'aten::get.default_Tensor(Dict(Tensor, t) self, Tensor key, t default_value) -> t(*)',
    'aten::get.float(Dict(float, t) self, float key) -> t(*)?',
    'aten::get.int(Dict(int, t) self, int key) -> t(*)?',
    'aten::get.str(Dict(str, t) self, str key) -> t(*)?',
    'aten::get.Tensor(Dict(Tensor, t) self, Tensor key) -> t(*)?',
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
    'aten::Int.bool(bool a) -> int',
    'aten::Int.float(float a) -> int',
    'aten::Int.Scalar(Scalar a) -> int',
    'aten::Int.str(str a) -> int',
    'aten::Int.Tensor(Tensor a) -> int',
    'aten::int_repr(Tensor self) -> Tensor',
    'aten::int_repr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::IntImplicit(Tensor a) -> int',
    'aten::inverse(Tensor self) -> Tensor',
    'aten::inverse.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)',
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
    'aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor',
    'aten::leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)',
    'aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result) -> Tensor',
    'aten::leaky_relu_backward.grad_input(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result, *, Tensor(a!) grad_input) -> Tensor(a!)',
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
    'aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor',
    'aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor',
    'aten::lerp.Tensor_out(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)',
    'aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)',
    'aten::less.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::less.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::less.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::less.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::less_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::less_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::less_equal.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::less_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::less_equal.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::less_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::less_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::less_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
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
    'aten::mul(Scalar a, Scalar b) -> Scalar',
    'aten::mul.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::mul.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::mul.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::mul.complex(complex a, complex b) -> complex',
    'aten::mul.complex_float(complex a, float b) -> complex',
    'aten::mul.complex_int(complex a, int b) -> complex',
    'aten::mul.float(float a, float b) -> float',
    'aten::mul.float_complex(float a, complex b) -> complex',
    'aten::mul.float_int(float a, int b) -> float',
    'aten::mul.int(int a, int b) -> int',
    'aten::mul.int_complex(int a, complex b) -> complex',
    'aten::mul.int_float(int a, float b) -> float',
    'aten::mul.left_t(t[] l, int n) -> t[]',
    'aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::mul.right_(int n, t[] l) -> t[]',
    'aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::mul_.t(t[](a!) l, int n) -> t[](a!)',
    'aten::ne(Scalar a, Scalar b) -> bool',
    'aten::ne.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::ne.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::ne.Tensor_list(Tensor[] a, Tensor[] b) -> bool',
    'aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::ne.bool(bool a, bool b) -> bool',
    'aten::ne.bool_list(bool[] a, bool[] b) -> bool',
    'aten::ne.complex(complex a, complex b) -> bool',
    'aten::ne.complex_float(complex a, float b) -> bool',
    'aten::ne.device(Device a, Device b) -> bool',
    'aten::ne.enum(AnyEnumType a, AnyEnumType b) -> bool',
    'aten::ne.float(float a, float b) -> bool',
    'aten::ne.float_complex(float a, complex b) -> bool',
    'aten::ne.float_int(float a, int b) -> bool',
    'aten::ne.float_list(float[] a, float[] b) -> bool',
    'aten::ne.int(int a, int b) -> bool',
    'aten::ne.int_float(int a, float b) -> bool',
    'aten::ne.int_list(int[] a, int[] b) -> bool',
    'aten::ne.str(str a, str b) -> bool',
    'aten::ne.str_list(str[] a, str[] b) -> bool',
    'aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::ne_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::neg(Tensor self) -> Tensor',
    'aten::neg.Scalar(Scalar a) -> Scalar',
    'aten::neg.complex(complex a) -> complex',
    'aten::neg.float(float a) -> float',
    'aten::neg.int(int a) -> int',
    'aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::neg_(Tensor(a!) self) -> Tensor(a!)',
    'aten::negative(Tensor self) -> Tensor',
    'aten::negative.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::negative_(Tensor(a!) self) -> Tensor(a!)',
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
    'aten::ScalarImplicit(Tensor a) -> Scalar',
    'aten::searchsorted.Scalar_out(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!)', 
    'aten::searchsorted.Scalar(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor', 
    'aten::searchsorted.Tensor_out(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!)', 
    'aten::searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor', 
    'aten::sqrt.complex(complex a) -> complex',
    'aten::sqrt.float(float a) -> float',
    'aten::sqrt.int(int a) -> float',
    'aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::sqrt.Scalar(Scalar a) -> Scalar',
    'aten::sqrt(Tensor self) -> Tensor',
    'aten::str(t elem) -> str',
    'aten::sub(Scalar a, Scalar b) -> Scalar',
    'aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor',
    'aten::sub.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor',
    'aten::sub.complex(complex a, complex b) -> complex',
    'aten::sub.complex_float(complex a, float b) -> complex',
    'aten::sub.complex_int(complex a, int b) -> complex',
    'aten::sub.float(float a, float b) -> float',
    'aten::sub.float_complex(float a, complex b) -> complex',
    'aten::sub.float_int(float a, int b) -> float',
    'aten::sub.int(int a, int b) -> int',
    'aten::sub.int_complex(int a, complex b) -> complex',
    'aten::sub.int_float(int a, float b) -> float',
    'aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)',
    'aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)',
    'aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)',
    'aten::subtract.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor',
    'aten::subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor',
    'aten::subtract.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)',
    'aten::subtract_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)',
    'aten::subtract_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)',
    'aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor',
    'aten::sum.DimnameList_out(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)',
    'aten::sum.IntList_out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)',
    'aten::sum.bool(bool[] self) -> int',
    'aten::sum.complex(complex[] self) -> complex',
    'aten::sum.dim_DimnameList(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor',
    'aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor',
    'aten::sum.float(float[] self) -> float',
    'aten::sum.int(int[] self) -> int',
    'aten::sum.out(Tensor self, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)',
    'aten::sum_to_size(Tensor self, SymInt[] size) -> Tensor',
    'aten::tensor(t[] data, *, ScalarType? dtype=None, Device? device=None, bool requires_grad=False) -> Tensor',
    'aten::tensor.bool(bool t, *, ScalarType? dtype=None, Device? device=None, bool requires_grad=False) -> Tensor',
    'aten::tensor.complex(complex t, *, ScalarType? dtype=None, Device? device=None, bool requires_grad=False) -> Tensor',
    'aten::tensor.float(float t, *, ScalarType? dtype=None, Device? device=None, bool requires_grad=False) -> Tensor',
    'aten::tensor.int(int t, *, ScalarType? dtype=None, Device? device=None, bool requires_grad=False) -> Tensor',
    'aten::tensor_split.indices(Tensor(a -> *) self, SymInt[] indices, int dim=0) -> Tensor(a)[]',
    'aten::tensor_split.sections(Tensor(a -> *) self, SymInt sections, int dim=0) -> Tensor(a)[]',
    'aten::tensor_split.tensor_indices_or_sections(Tensor(a -> *) self, Tensor tensor_indices_or_sections, int dim=0) -> Tensor(a)[]',
    'aten::tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor',
    'aten::tensordot.out(Tensor self, Tensor other, int[] dims_self, int[] dims_other, *, Tensor(a!) out) -> Tensor(a!)',
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

known_legacy_schema_definitions = [
    '_caffe2::BBoxTransform(Tensor rois, Tensor deltas, Tensor im_info, float[] weights, bool apply_scale, bool rotated, bool angle_bound_on, int angle_bound_lo, int angle_bound_hi, float clip_angle_thresh, bool legacy_plus_one) -> (Tensor output_0, Tensor output_1)',
    '_caffe2::BatchPermutation(Tensor X, Tensor indices) -> Tensor',
    '_caffe2::BoxWithNMSLimit(Tensor scores, Tensor boxes, Tensor batch_splits, float score_thresh, float nms, int detections_per_im, bool soft_nms_enabled, str soft_nms_method, float soft_nms_sigma, float soft_nms_min_score_thres, bool rotated, bool cls_agnostic_bbox_reg, bool input_boxes_include_bg_cls, bool output_classes_include_bg_cls, bool legacy_plus_one) -> (Tensor scores, Tensor boxes, Tensor classes, Tensor batch_splits, Tensor keeps, Tensor keeps_size)',
    '_caffe2::CollectAndDistributeFpnRpnProposals(Tensor[] input_list, int roi_canonical_scale, int roi_canonical_level, int roi_max_level, int roi_min_level, int rpn_max_level, int rpn_min_level, int rpn_post_nms_topN, bool legacy_plus_one) -> (Tensor rois, Tensor rois_fpn2, Tensor rois_fpn3, Tensor rois_fpn4, Tensor rois_fpn5, Tensor rois_idx_restore_int32)',
    '_caffe2::CollectRpnProposals(Tensor[] input_list, int rpn_max_level, int rpn_min_level, int rpn_post_nms_topN) -> (Tensor rois)',
    '_caffe2::CopyCPUToGPU(Tensor input) -> Tensor',
    '_caffe2::CopyGPUToCPU(Tensor input) -> Tensor',
    '_caffe2::DistributeFpnProposals(Tensor rois, int roi_canonical_scale, int roi_canonical_level, int roi_max_level, int roi_min_level, bool legacy_plus_one) -> (Tensor rois_fpn2, Tensor rois_fpn3, Tensor rois_fpn4, Tensor rois_fpn5, Tensor rois_idx_restore_int32)',
    '_caffe2::GenerateProposals(Tensor scores, Tensor bbox_deltas, Tensor im_info, Tensor anchors, float spatial_scale, int pre_nms_topN, int post_nms_topN, float nms_thresh, float min_size, bool angle_bound_on, int angle_bound_lo, int angle_bound_hi, float clip_angle_thresh, bool legacy_plus_one) -> (Tensor output_0, Tensor output_1)',
    '_caffe2::RoIAlign(Tensor features, Tensor rois, str order, float spatial_scale, int pooled_h, int pooled_w, int sampling_ratio, bool aligned) -> Tensor',
    'aten::arange.start_out_(Scalar start, Scalar end) -> Tensor',
    'aten::fft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor',
    'aten::grid_sampler.legacy(Tensor input, Tensor grid, int interpolation_mode, int padding_mode) -> Tensor',
    'neuron::forward_v2_1(Tensor[] _0, __torch__.torch.classes.neuron.Model _1) -> (Tensor _0)',
    'prim::shape(Tensor self) -> int[]',
    'torchaudio::sox_effects_apply_effects_tensor(Tensor tensor, int sample_rate, str[][] effects, bool channels_first=True) -> (Tensor, int64)',
    'torchvision::nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor',
    'torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio, bool aligned) -> Tensor',
]

# pylint: enable=line-too-long

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
    for value in known_legacy_schema_definitions:
        schema = pytorch.Schema(value)
        schemas[schema.name] = schema
    for value in known_schema_definitions:
        schema = pytorch.Schema(value)
        schemas[schema.name] = schema
    return schemas

def _filter_schemas(schemas, types):
    keys = set(map(lambda _: _.split('.')[0], types.keys()))
    filtered_schemas = set()
    for schema in schemas.values():
        for key in keys:
            if schema.name == key or schema.name.startswith(key + '.'):
                filtered_schemas.add(schema.name)
    # for schema in schemas.values():
    #    if schema.name.startswith('aten::pop'):
    #         filtered_schemas.add(schema.name)
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
        'aten::classes._nnapi.Compilation'
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
        value = metadata.type(schema)
        value['name'] = schema.value
    _write_metadata(types)

def main(): # pylint: disable=missing-function-docstring
    _metadata()

if __name__ == '__main__':
    main()
