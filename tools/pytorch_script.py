''' TorchScript metadata script '''
# pylint: disable=too-many-lines

import collections
import json
import os
import re
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.pycache_prefix = os.path.join(root_dir, 'dist', 'pycache', 'pytorch_script')

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

def _write_metadata(metadata):
    content = json.dumps(metadata, indent=2, ensure_ascii=False)
    content = re.sub(r'\s {8}', ' ', content)
    content = re.sub(r',\s {8}', ', ', content)
    content = re.sub(r'\s {6}}', ' }', content)
    _write(metadata_file, content)

known_legacy_schema_definitions = [
    # pylint: disable=line-too-long
    '_caffe2::BBoxTransform(Tensor rois, Tensor deltas, Tensor im_info, float[] weights, bool apply_scale, bool rotated, bool angle_bound_on, int angle_bound_lo, int angle_bound_hi, float clip_angle_thresh, bool legacy_plus_one, Tensor[]? _caffe2_preallocated_outputs=None) -> (Tensor output_0, Tensor output_1)',
    '_caffe2::BatchPermutation(Tensor X, Tensor indices, Tensor[]? _caffe2_preallocated_outputs=None) -> Tensor',
    '_caffe2::BoxWithNMSLimit(Tensor scores, Tensor boxes, Tensor batch_splits, float score_thresh, float nms, int detections_per_im, bool soft_nms_enabled, str soft_nms_method, float soft_nms_sigma, float soft_nms_min_score_thres, bool rotated, bool cls_agnostic_bbox_reg, bool input_boxes_include_bg_cls, bool output_classes_include_bg_cls, bool legacy_plus_one, Tensor[]? _caffe2_preallocated_outputs=None) -> (Tensor scores, Tensor boxes, Tensor classes, Tensor batch_splits, Tensor keeps, Tensor keeps_size)',
    '_caffe2::CollectAndDistributeFpnRpnProposals(Tensor[] input_list, int roi_canonical_scale, int roi_canonical_level, int roi_max_level, int roi_min_level, int rpn_max_level, int rpn_min_level, int rpn_post_nms_topN, bool legacy_plus_one, Tensor[]? _caffe2_preallocated_outputs=None) -> (Tensor rois, Tensor rois_fpn2, Tensor rois_fpn3, Tensor rois_fpn4, Tensor rois_fpn5, Tensor rois_idx_restore_int32)',
    '_caffe2::CollectRpnProposals(Tensor[] input_list, int rpn_max_level, int rpn_min_level, int rpn_post_nms_topN, Tensor[]? _caffe2_preallocated_outputs=None) -> (Tensor rois)',
    '_caffe2::CopyCPUToGPU(Tensor input, Tensor[]? _caffe2_preallocated_outputs=None) -> Tensor',
    '_caffe2::CopyGPUToCPU(Tensor input, Tensor[]? _caffe2_preallocated_outputs=None) -> Tensor',
    '_caffe2::DistributeFpnProposals(Tensor rois, int roi_canonical_scale, int roi_canonical_level, int roi_max_level, int roi_min_level, bool legacy_plus_one, Tensor[]? _caffe2_preallocated_outputs=None) -> (Tensor rois_fpn2, Tensor rois_fpn3, Tensor rois_fpn4, Tensor rois_fpn5, Tensor rois_idx_restore_int32)',
    '_caffe2::GenerateProposals(Tensor scores, Tensor bbox_deltas, Tensor im_info, Tensor anchors, float spatial_scale, int pre_nms_topN, int post_nms_topN, float nms_thresh, float min_size, bool angle_bound_on, int angle_bound_lo, int angle_bound_hi, float clip_angle_thresh, bool legacy_plus_one, Tensor[]? _caffe2_preallocated_outputs=None) -> (Tensor output_0, Tensor output_1)',
    '_caffe2::RoIAlign(Tensor features, Tensor rois, str order, float spatial_scale, int pooled_h, int pooled_w, int sampling_ratio, bool aligned, Tensor[]? _caffe2_preallocated_outputs=None) -> Tensor',
    'aten::_cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::_cat(Tensor[] tensors, int dim=0) -> Tensor',
    'aten::arange.start_out_(Scalar start, Scalar end) -> Tensor',
    'aten::fft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor',
    'aten::grid_sampler.legacy(Tensor input, Tensor grid, int interpolation_mode, int padding_mode) -> Tensor',
    'neuron::forward_v2_1(Tensor[] _0, __torch__.torch.classes.neuron.Model _1) -> (Tensor _0)',
    'prim::isinstance(Any to_check) -> bool',
    'prim::shape(Tensor self) -> int[]',
    'quantized_decomposed::quantize_per_tensor.out(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)',
    'quantized_decomposed::dequantize_per_tensor.out(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None, Tensor(a!) out) -> Tensor(a!)',
    'quantized_decomposed::dequantize_per_tensor.Tensor_out(Tensor input, Tensor scale, Tensor zero_point, int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None, Tensor(a!) out) -> Tensor(a!)',
    'torch_scatter::gather_coo(Tensor _0, Tensor _1, Tensor? _2) -> Tensor _0',
    'torch_scatter::segment_max_coo(Tensor _0, Tensor _1, Tensor? _2, int? _3) -> (Tensor _0, Tensor _1)',
    'torch_scatter::segment_min_coo(Tensor _0, Tensor _1, Tensor? _2, int? _3) -> (Tensor _0, Tensor _1)',
    'torchaudio::sox_effects_apply_effects_tensor(Tensor tensor, int sample_rate, str[][] effects, bool channels_first=True) -> (Tensor, int)',
    'torch_scatter::segment_mean_coo(Tensor _0, Tensor _1, Tensor? _2, int? _3) -> Tensor _0',
    'torch_scatter::segment_sum_coo(Tensor _0, Tensor _1, Tensor? _2, int? _3) -> Tensor _0',
    'torch_scatter::gather_csr(Tensor _0, Tensor _1, Tensor? _2) -> Tensor _0',
    'torch_scatter::segment_max_csr(Tensor _0, Tensor _1, Tensor? _2) -> (Tensor _0, Tensor _1)',
    'torch_scatter::segment_min_csr(Tensor _0, Tensor _1, Tensor? _2) -> (Tensor _0, Tensor _1)',
    'torch_scatter::segment_mean_csr(Tensor _0, Tensor _1, Tensor? _2) -> Tensor _0',
    'torch_scatter::segment_sum_csr(Tensor _0, Tensor _1, Tensor? _2) -> Tensor _0',
    'torch_scatter::scatter_max(Tensor _0, Tensor _1, int _2, Tensor? _3, int? _4) -> (Tensor _0, Tensor _1)',
    'torch_scatter::scatter_min(Tensor _0, Tensor _1, int _2, Tensor? _3, int? _4) -> (Tensor _0, Tensor _1)',
    'torch_scatter::scatter_mean(Tensor _0, Tensor _1, int _2, Tensor? _3, int? _4) -> Tensor _0',
    'torch_scatter::scatter_mul(Tensor _0, Tensor _1, int _2, Tensor? _3, int? _4) -> Tensor _0',
    'torch_scatter::scatter_sum(Tensor _0, Tensor _1, int _2, Tensor? _3, int? _4) -> Tensor _0',
    'torch_scatter::cuda_version() -> int _0',
    # pylint: enable=line-too-long
]

def _identifier(schema):
    return schema.split('(', 1)[0].strip()

def _all_schemas():
    torch = __import__('torch')
    __import__('torchvision')
    __import__('torchaudio')
    return list(torch._C._jit_get_all_schemas()) # pylint: disable=protected-access

def _parse_schemas():
    schemas = {}
    for schema in _all_schemas():
        definition = str(schema)
        definition = definition.replace('(b|a)', '(a|b)')
        key = _identifier(definition)
        schemas[key] = definition
    for schema in known_legacy_schema_definitions:
        key = _identifier(schema)
        if key not in schemas:
            schemas[key] = schema
        else:
            print(f'-> {key}')
    return schemas

def _filter_schemas(schemas, types):
    names = set(map(lambda _: _.split('.')[0], types.keys()))
    for key in known_legacy_schema_definitions:
        names.add(re.sub(r'[\.(].*$', '', key))
    filtered_schemas = set()
    for schema in schemas.values():
        for name in names:
            key = _identifier(schema)
            if key == name or key.startswith(name + '.'):
                filtered_schemas.add(key)
    return dict(filter(lambda _: _[0] in filtered_schemas, schemas.items()))

def _check_types(types, schemas):
    types = dict(types.items())
    for schema in schemas.values():
        key = _identifier(schema)
        if key in types:
            types.pop(key)
    for key in list(types.keys()):
        if key.startswith('torch.nn') or key.startswith('__torch__.'):
            types.pop(key)
    if len(types) > 0:
        raise Exception('\n'.join(list(types.keys()))) # pylint: disable=broad-exception-raised

def _sort_types(types):
    keys = {}
    index = 0
    for schema in _all_schemas():
        definition = str(schema)
        key = _identifier(definition)
        keys[key] = index
        index += 1
    classes = collections.OrderedDict()
    for item in types:
        name = item['name']
        if name.find('::') == -1:
            classes[name] = item
        else:
            key = _identifier(name)
            if not key in keys:
                keys[key] = index
                index += 1
    for key, _ in classes.items():
        keys[key] = index
        index += 1
    def custom_key(x):
        key = _identifier(x['name'])
        return keys[key]
    types = sorted(types, key=custom_key)
    return types


def _metadata():
    types = _read_metadata()
    schemas = _parse_schemas()
    _check_types(types, schemas)
    filtered_schemas = _filter_schemas(schemas, types)
    for schema in filtered_schemas.values():
        key = _identifier(schema)
        if key in types:
            types[key]['name'] = schema
        else:
            types[key] = { 'name': schema }
    types = _sort_types(list(types.values()))
    _write_metadata(types)

def main(): # pylint: disable=missing-function-docstring
    _metadata()

if __name__ == '__main__':
    main()
