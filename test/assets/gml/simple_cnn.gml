Creator "Netron GML Example"
Version "1.0"
graph [
  directed 1
  node [
    id 0
    label "input"
    op_type "Input"
    shape "1x3x224x224"
    dtype "float32"
  ]
  node [
    id 1
    label "conv1"
    op_type "Conv"
    kernel_shape "3x3"
    strides "1x1"
    pads "1x1x1x1"
    out_channels 64
  ]
  node [
    id 2
    label "bn1"
    op_type "BatchNormalization"
    epsilon 0.00001
  ]
  node [
    id 3
    label "relu1"
    op_type "Relu"
  ]
  node [
    id 4
    label "pool1"
    op_type "MaxPool"
    kernel_shape "2x2"
    strides "2x2"
  ]
  node [
    id 5
    label "flatten"
    op_type "Flatten"
  ]
  node [
    id 6
    label "fc1"
    op_type "Gemm"
    out_features 10
  ]
  node [
    id 7
    label "softmax"
    op_type "Softmax"
  ]
  node [
    id 8
    label "output"
    op_type "Output"
  ]
  edge [
    source 0
    target 1
  ]
  edge [
    source 1
    target 2
  ]
  edge [
    source 2
    target 3
  ]
  edge [
    source 3
    target 4
  ]
  edge [
    source 4
    target 5
  ]
  edge [
    source 5
    target 6
  ]
  edge [
    source 6
    target 7
  ]
  edge [
    source 7
    target 8
  ]
]
