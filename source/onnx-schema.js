var $root = flatbuffers.get('ort');

$root.onnxruntime = $root.onnxruntime || {};

$root.onnxruntime.experimental = $root.onnxruntime.experimental || {};

$root.onnxruntime.experimental.fbs = $root.onnxruntime.experimental.fbs || {};

$root.onnxruntime.experimental.fbs.AttributeType = {
    UNDEFINED: 0,
    FLOAT: 1,
    INT: 2,
    STRING: 3,
    TENSOR: 4,
    GRAPH: 5,
    FLOATS: 6,
    INTS: 7,
    STRINGS: 8,
    TENSORS: 9,
    GRAPHS: 10,
    SPARSE_TENSOR: 11,
    SPARSE_TENSORS: 12
};

$root.onnxruntime.experimental.fbs.Shape = class Shape {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.Shape();
        $.dim = reader.tableArray(position, 4, $root.onnxruntime.experimental.fbs.Dimension.decode);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.Dimension = class Dimension {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.Dimension();
        $.value = reader.table(position, 4, $root.onnxruntime.experimental.fbs.DimensionValue.decode);
        $.denotation = reader.string_(position, 6, null);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.DimensionValueType = {
    UNKNOWN: 0,
    VALUE: 1,
    PARAM: 2
};

$root.onnxruntime.experimental.fbs.DimensionValue = class DimensionValue {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.DimensionValue();
        $.dim_type = reader.int8_(position, 4, 0);
        $.dim_value = reader.int64_(position, 6, 0);
        $.dim_param = reader.string_(position, 8, null);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.TensorDataType = {
    UNDEFINED: 0,
    FLOAT: 1,
    UINT8: 2,
    INT8: 3,
    UINT16: 4,
    INT16: 5,
    INT32: 6,
    INT64: 7,
    STRING: 8,
    BOOL: 9,
    FLOAT16: 10,
    DOUBLE: 11,
    UINT32: 12,
    UINT64: 13,
    COMPLEX64: 14,
    COMPLEX128: 15,
    BFLOAT16: 16
};

$root.onnxruntime.experimental.fbs.TensorTypeAndShape = class TensorTypeAndShape {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.TensorTypeAndShape();
        $.elem_type = reader.int32_(position, 4, 0);
        $.shape = reader.table(position, 6, $root.onnxruntime.experimental.fbs.Shape.decode);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.MapType = class MapType {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.MapType();
        $.key_type = reader.int32_(position, 4, 0);
        $.value_type = reader.table(position, 6, $root.onnxruntime.experimental.fbs.TypeInfo.decode);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.SequenceType = class SequenceType {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.SequenceType();
        $.elem_type = reader.table(position, 4, $root.onnxruntime.experimental.fbs.TypeInfo.decode);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.NodeType = {
    Primitive: 0,
    Fused: 1
};

$root.onnxruntime.experimental.fbs.EdgeEnd = class EdgeEnd {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.EdgeEnd();
        $.node_index = reader.uint32(position + 0);
        $.src_arg_index = reader.int32(position + 4);
        $.dst_arg_index = reader.int32(position + 8);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.NodeEdge = class NodeEdge {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.NodeEdge();
        $.node_index = reader.uint32_(position, 4, 0);
        $.input_edges = reader.structArray(position, 6, undefined,$root.onnxruntime.experimental.fbs.EdgeEnd.decode);
        $.output_edges = reader.structArray(position, 8, undefined,$root.onnxruntime.experimental.fbs.EdgeEnd.decode);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.Node = class Node {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.Node();
        $.name = reader.string_(position, 4, null);
        $.doc_string = reader.string_(position, 6, null);
        $.domain = reader.string_(position, 8, null);
        $.since_version = reader.int32_(position, 10, 0);
        $.index = reader.uint32_(position, 12, 0);
        $.op_type = reader.string_(position, 14, null);
        $.type = reader.int32_(position, 16, 0);
        $.execution_provider_type = reader.string_(position, 18, null);
        $.inputs = reader.strings_(position, 20);
        $.outputs = reader.strings_(position, 22);
        $.attributes = reader.tableArray(position, 24, $root.onnxruntime.experimental.fbs.Attribute.decode);
        $.input_arg_counts = reader.typedArray(position, 26, Int32Array);
        $.implicit_inputs = reader.strings_(position, 28);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.ValueInfo = class ValueInfo {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.ValueInfo();
        $.name = reader.string_(position, 4, null);
        $.doc_string = reader.string_(position, 6, null);
        $.type = reader.table(position, 8, $root.onnxruntime.experimental.fbs.TypeInfo.decode);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.TypeInfoValue = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return $root.onnxruntime.experimental.fbs.TensorTypeAndShape.decode(reader, position);
            case 2: return $root.onnxruntime.experimental.fbs.SequenceType.decode(reader, position);
            case 3: return $root.onnxruntime.experimental.fbs.MapType.decode(reader, position);
        }
        return undefined;
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'TensorTypeAndShape': return $root.onnxruntime.experimental.fbs.TensorTypeAndShape.decodeText(reader, json);
            case 'SequenceType': return $root.onnxruntime.experimental.fbs.SequenceType.decodeText(reader, json);
            case 'MapType': return $root.onnxruntime.experimental.fbs.MapType.decodeText(reader, json);
        }
        return undefined;
    }
};

$root.onnxruntime.experimental.fbs.TypeInfo = class TypeInfo {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.TypeInfo();
        $.denotation = reader.string_(position, 4, null);
        $.value = reader.union(position, 6, $root.onnxruntime.experimental.fbs.TypeInfoValue.decode);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.OperatorSetId = class OperatorSetId {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.OperatorSetId();
        $.domain = reader.string_(position, 4, null);
        $.version = reader.int64_(position, 6, 0);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.Tensor();
        $.name = reader.string_(position, 4, null);
        $.doc_string = reader.string_(position, 6, null);
        $.dims = reader.int64s_(position, 8);
        $.data_type = reader.int32_(position, 10, 0);
        $.raw_data = reader.typedArray(position, 12, Uint8Array);
        $.string_data = reader.strings_(position, 14);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.SparseTensor = class SparseTensor {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.SparseTensor();
        $.values = reader.table(position, 4, $root.onnxruntime.experimental.fbs.Tensor.decode);
        $.indices = reader.table(position, 6, $root.onnxruntime.experimental.fbs.Tensor.decode);
        $.dims = reader.int64s_(position, 8);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.Attribute = class Attribute {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.Attribute();
        $.name = reader.string_(position, 4, null);
        $.doc_string = reader.string_(position, 6, null);
        $.type = reader.int32_(position, 8, 0);
        $.f = reader.float32_(position, 10, 0);
        $.i = reader.int64_(position, 12, 0);
        $.s = reader.string_(position, 14, null);
        $.t = reader.table(position, 16, $root.onnxruntime.experimental.fbs.Tensor.decode);
        $.g = reader.table(position, 18, $root.onnxruntime.experimental.fbs.Graph.decode);
        $.floats = reader.typedArray(position, 20, Float32Array);
        $.ints = reader.int64s_(position, 22);
        $.strings = reader.strings_(position, 24);
        $.tensors = reader.tableArray(position, 26, $root.onnxruntime.experimental.fbs.Tensor.decode);
        $.graphs = reader.tableArray(position, 28, $root.onnxruntime.experimental.fbs.Graph.decode);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.Graph = class Graph {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.Graph();
        $.initializers = reader.tableArray(position, 4, $root.onnxruntime.experimental.fbs.Tensor.decode);
        $.node_args = reader.tableArray(position, 6, $root.onnxruntime.experimental.fbs.ValueInfo.decode);
        $.nodes = reader.tableArray(position, 8, $root.onnxruntime.experimental.fbs.Node.decode);
        $.max_node_index = reader.uint32_(position, 10, 0);
        $.node_edges = reader.tableArray(position, 12, $root.onnxruntime.experimental.fbs.NodeEdge.decode);
        $.inputs = reader.strings_(position, 14);
        $.outputs = reader.strings_(position, 16);
        $.sparse_initializers = reader.tableArray(position, 18, $root.onnxruntime.experimental.fbs.SparseTensor.decode);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.StringStringEntry = class StringStringEntry {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.StringStringEntry();
        $.key = reader.string_(position, 4, null);
        $.value = reader.string_(position, 6, null);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.Model = class Model {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.Model();
        $.ir_version = reader.int64_(position, 4, 0);
        $.opset_import = reader.tableArray(position, 6, $root.onnxruntime.experimental.fbs.OperatorSetId.decode);
        $.producer_name = reader.string_(position, 8, null);
        $.producer_version = reader.string_(position, 10, null);
        $.domain = reader.string_(position, 12, null);
        $.model_version = reader.int64_(position, 14, 0);
        $.doc_string = reader.string_(position, 16, null);
        $.graph = reader.table(position, 18, $root.onnxruntime.experimental.fbs.Graph.decode);
        $.graph_doc_string = reader.string_(position, 20, null);
        $.metadata_props = reader.tableArray(position, 22, $root.onnxruntime.experimental.fbs.StringStringEntry.decode);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.KernelCreateInfos = class KernelCreateInfos {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.KernelCreateInfos();
        $.node_indices = reader.typedArray(position, 4, Uint32Array);
        $.kernel_def_hashes = reader.uint64s_(position, 6);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.SubGraphSessionState = class SubGraphSessionState {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.SubGraphSessionState();
        $.graph_id = reader.string_(position, 4, null);
        $.session_state = reader.table(position, 6, $root.onnxruntime.experimental.fbs.SessionState.decode);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.SessionState = class SessionState {

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.SessionState();
        $.kernels = reader.table(position, 4, $root.onnxruntime.experimental.fbs.KernelCreateInfos.decode);
        $.sub_graph_session_states = reader.tableArray(position, 6, $root.onnxruntime.experimental.fbs.SubGraphSessionState.decode);
        return $;
    }
};

$root.onnxruntime.experimental.fbs.InferenceSession = class InferenceSession {

    static identifier(reader) {
        return reader.identifier === 'ORTM';
    }

    static create(reader) {
        return $root.onnxruntime.experimental.fbs.InferenceSession.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new $root.onnxruntime.experimental.fbs.InferenceSession();
        $.ort_version = reader.string_(position, 4, null);
        $.model = reader.table(position, 6, $root.onnxruntime.experimental.fbs.Model.decode);
        $.session_state = reader.table(position, 8, $root.onnxruntime.experimental.fbs.SessionState.decode);
        return $;
    }
};
