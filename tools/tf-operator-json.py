#!/usr/bin/env python

from tensorflow.core.framework import op_def_pb2
from google.protobuf import text_format

input_file = '../third_party/tensorflow/tensorflow/core/ops/ops.pbtxt';
output_file = '../src/tf-operator.pb'

with open(input_file) as input_handle:
    ops_list = op_def_pb2.OpList()
    text_format.Merge(input_handle.read(), ops_list)
    data = ops_list.SerializeToString()
    with open(output_file, 'wb') as output_handle:
        output_handle.write(data)
