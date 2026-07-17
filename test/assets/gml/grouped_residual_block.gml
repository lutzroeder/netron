Creator "Netron GML Example"
Version "1.0"
graph [
  directed 1
  node [
    id 100
    isGroup 1
    label "ResidualBlock"
  ]
  node [
    id 0
    label "input"
    op_type "Input"
  ]
  node [
    id 1
    label "conv_a"
    op_type "Conv"
    gid 100
    kernel_shape "3x3"
  ]
  node [
    id 2
    label "relu_a"
    op_type "Relu"
    gid 100
  ]
  node [
    id 3
    label "conv_b"
    op_type "Conv"
    gid 100
    kernel_shape "3x3"
  ]
  node [
    id 4
    label "weight_const"
    is_buffer 1
    shape "64x64x3x3"
    dtype "float32"
  ]
  node [
    id 5
    label "add"
    op_type "Add"
  ]
  node [
    id 6
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
    source 4
    target 3
  ]
  edge [
    source 0
    target 5
  ]
  edge [
    source 3
    target 5
  ]
  edge [
    source 5
    target 6
  ]
]
