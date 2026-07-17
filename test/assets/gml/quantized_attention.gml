Creator "Netron GML Example"
Version "1.0"
graph [
  directed 1
  node [
    id 0
    label "input"
    op_type "Input"
    shape "1x128x768"
    dtype "float32"
  ]
  node [
    id 1
    label "layernorm1"
    op_type "LayerNormalization"
    epsilon 0.000001
  ]
  node [
    id 2
    label "self_attention"
    op_type "Attention"
    num_heads 12
    head_dim 64
  ]
  node [
    id 3
    label "quant1"
    op_type "QuantizeLinear"
    scale 0.0078125
    zero_point 128
  ]
  node [
    id 4
    label "dequant1"
    op_type "DequantizeLinear"
  ]
  node [
    id 5
    label "layernorm2"
    op_type "LayerNormalization"
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
]
