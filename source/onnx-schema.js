
export const onnxruntime = {};

onnxruntime.fbs = onnxruntime.fbs || {};

onnxruntime.fbs.AttributeType = {
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

onnxruntime.fbs.Shape = class Shape {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.Shape();
        $.dim = reader.tableArray(position, 4, onnxruntime.fbs.Dimension.decode);
        return $;
    }
};

onnxruntime.fbs.Dimension = class Dimension {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.Dimension();
        $.value = reader.table(position, 4, onnxruntime.fbs.DimensionValue.decode);
        $.denotation = reader.string_(position, 6, null);
        return $;
    }
};

onnxruntime.fbs.DimensionValueType = {
    UNKNOWN: 0,
    VALUE: 1,
    PARAM: 2
};

onnxruntime.fbs.DimensionValue = class DimensionValue {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.DimensionValue();
        $.dim_type = reader.int8_(position, 4, 0);
        $.dim_value = reader.int64_(position, 6, 0);
        $.dim_param = reader.string_(position, 8, null);
        return $;
    }
};

onnxruntime.fbs.TensorDataType = {
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
    BFLOAT16: 16,
    FLOAT8E4M3FN: 17,
    FLOAT8E4M3FNUZ: 18,
    FLOAT8E5M2: 19,
    FLOAT8E5M2FNUZ: 20
};

onnxruntime.fbs.TensorTypeAndShape = class TensorTypeAndShape {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.TensorTypeAndShape();
        $.elem_type = reader.int32_(position, 4, 0);
        $.shape = reader.table(position, 6, onnxruntime.fbs.Shape.decode);
        return $;
    }
};

onnxruntime.fbs.MapType = class MapType {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.MapType();
        $.key_type = reader.int32_(position, 4, 0);
        $.value_type = reader.table(position, 6, onnxruntime.fbs.TypeInfo.decode);
        return $;
    }
};

onnxruntime.fbs.SequenceType = class SequenceType {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.SequenceType();
        $.elem_type = reader.table(position, 4, onnxruntime.fbs.TypeInfo.decode);
        return $;
    }
};

onnxruntime.fbs.NodeType = {
    Primitive: 0,
    Fused: 1
};

onnxruntime.fbs.EdgeEnd = class EdgeEnd {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.EdgeEnd();
        $.node_index = reader.uint32(position + 0);
        $.src_arg_index = reader.int32(position + 4);
        $.dst_arg_index = reader.int32(position + 8);
        return $;
    }
};

onnxruntime.fbs.NodeEdge = class NodeEdge {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.NodeEdge();
        $.node_index = reader.uint32_(position, 4, 0);
        $.input_edges = reader.structArray(position, 6, onnxruntime.fbs.EdgeEnd.decode);
        $.output_edges = reader.structArray(position, 8, onnxruntime.fbs.EdgeEnd.decode);
        return $;
    }
};

onnxruntime.fbs.Node = class Node {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.Node();
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
        $.attributes = reader.tableArray(position, 24, onnxruntime.fbs.Attribute.decode);
        $.input_arg_counts = reader.typedArray(position, 26, Int32Array);
        $.implicit_inputs = reader.strings_(position, 28);
        return $;
    }
};

onnxruntime.fbs.ValueInfo = class ValueInfo {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.ValueInfo();
        $.name = reader.string_(position, 4, null);
        $.doc_string = reader.string_(position, 6, null);
        $.type = reader.table(position, 8, onnxruntime.fbs.TypeInfo.decode);
        return $;
    }
};

onnxruntime.fbs.TypeInfoValue = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return onnxruntime.fbs.TensorTypeAndShape.decode(reader, position);
            case 2: return onnxruntime.fbs.SequenceType.decode(reader, position);
            case 3: return onnxruntime.fbs.MapType.decode(reader, position);
            default: return undefined;
        }
    }
};

onnxruntime.fbs.TypeInfo = class TypeInfo {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.TypeInfo();
        $.denotation = reader.string_(position, 4, null);
        $.value = reader.union(position, 6, onnxruntime.fbs.TypeInfoValue.decode);
        return $;
    }
};

onnxruntime.fbs.OperatorSetId = class OperatorSetId {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.OperatorSetId();
        $.domain = reader.string_(position, 4, null);
        $.version = reader.int64_(position, 6, 0);
        return $;
    }
};

onnxruntime.fbs.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.Tensor();
        $.name = reader.string_(position, 4, null);
        $.doc_string = reader.string_(position, 6, null);
        $.dims = reader.int64s_(position, 8);
        $.data_type = reader.int32_(position, 10, 0);
        $.raw_data = reader.typedArray(position, 12, Uint8Array);
        $.string_data = reader.strings_(position, 14);
        return $;
    }
};

onnxruntime.fbs.SparseTensor = class SparseTensor {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.SparseTensor();
        $.values = reader.table(position, 4, onnxruntime.fbs.Tensor.decode);
        $.indices = reader.table(position, 6, onnxruntime.fbs.Tensor.decode);
        $.dims = reader.int64s_(position, 8);
        return $;
    }
};

onnxruntime.fbs.Attribute = class Attribute {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.Attribute();
        $.name = reader.string_(position, 4, null);
        $.doc_string = reader.string_(position, 6, null);
        $.type = reader.int32_(position, 8, 0);
        $.f = reader.float32_(position, 10, 0);
        $.i = reader.int64_(position, 12, 0);
        $.s = reader.string_(position, 14, null);
        $.t = reader.table(position, 16, onnxruntime.fbs.Tensor.decode);
        $.g = reader.table(position, 18, onnxruntime.fbs.Graph.decode);
        $.floats = reader.typedArray(position, 20, Float32Array);
        $.ints = reader.int64s_(position, 22);
        $.strings = reader.strings_(position, 24);
        $.tensors = reader.tableArray(position, 26, onnxruntime.fbs.Tensor.decode);
        $.graphs = reader.tableArray(position, 28, onnxruntime.fbs.Graph.decode);
        return $;
    }
};

onnxruntime.fbs.NodesToOptimizeIndices = class NodesToOptimizeIndices {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.NodesToOptimizeIndices();
        $.node_indices = reader.typedArray(position, 4, Uint32Array);
        $.num_inputs = reader.uint32_(position, 6, 0);
        $.num_outputs = reader.uint32_(position, 8, 0);
        $.has_variadic_input = reader.bool_(position, 10, false);
        $.has_variadic_output = reader.bool_(position, 12, false);
        $.num_variadic_inputs = reader.uint32_(position, 14, 0);
        $.num_variadic_outputs = reader.uint32_(position, 16, 0);
        return $;
    }
};

onnxruntime.fbs.DeprecatedNodeIndexAndKernelDefHash = class DeprecatedNodeIndexAndKernelDefHash {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.DeprecatedNodeIndexAndKernelDefHash();
        $.node_index = reader.uint32_(position, 4, 0);
        $.kernel_def_hash = reader.uint64_(position, 6, 0);
        return $;
    }
};

onnxruntime.fbs.RuntimeOptimizationRecord = class RuntimeOptimizationRecord {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.RuntimeOptimizationRecord();
        $.action_id = reader.string_(position, 4, null);
        $.nodes_to_optimize_indices = reader.table(position, 6, onnxruntime.fbs.NodesToOptimizeIndices.decode);
        $.produced_nodes = reader.tableArray(position, 8, onnxruntime.fbs.DeprecatedNodeIndexAndKernelDefHash.decode);
        $.produced_op_ids = reader.strings_(position, 10);
        return $;
    }
};

onnxruntime.fbs.RuntimeOptimizationRecordContainerEntry = class RuntimeOptimizationRecordContainerEntry {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.RuntimeOptimizationRecordContainerEntry();
        $.optimizer_name = reader.string_(position, 4, null);
        $.runtime_optimization_records = reader.tableArray(position, 6, onnxruntime.fbs.RuntimeOptimizationRecord.decode);
        return $;
    }
};

onnxruntime.fbs.RuntimeOptimizations = class RuntimeOptimizations {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.RuntimeOptimizations();
        $.records = reader.tableArray(position, 4, onnxruntime.fbs.RuntimeOptimizationRecordContainerEntry.decode);
        return $;
    }
};

onnxruntime.fbs.Graph = class Graph {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.Graph();
        $.initializers = reader.tableArray(position, 4, onnxruntime.fbs.Tensor.decode);
        $.node_args = reader.tableArray(position, 6, onnxruntime.fbs.ValueInfo.decode);
        $.nodes = reader.tableArray(position, 8, onnxruntime.fbs.Node.decode);
        $.max_node_index = reader.uint32_(position, 10, 0);
        $.node_edges = reader.tableArray(position, 12, onnxruntime.fbs.NodeEdge.decode);
        $.inputs = reader.strings_(position, 14);
        $.outputs = reader.strings_(position, 16);
        $.sparse_initializers = reader.tableArray(position, 18, onnxruntime.fbs.SparseTensor.decode);
        $.runtime_optimizations = reader.table(position, 20, onnxruntime.fbs.RuntimeOptimizations.decode);
        return $;
    }
};

onnxruntime.fbs.StringStringEntry = class StringStringEntry {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.StringStringEntry();
        $.key = reader.string_(position, 4, null);
        $.value = reader.string_(position, 6, null);
        return $;
    }
};

onnxruntime.fbs.Model = class Model {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.Model();
        $.ir_version = reader.int64_(position, 4, 0);
        $.opset_import = reader.tableArray(position, 6, onnxruntime.fbs.OperatorSetId.decode);
        $.producer_name = reader.string_(position, 8, null);
        $.producer_version = reader.string_(position, 10, null);
        $.domain = reader.string_(position, 12, null);
        $.model_version = reader.int64_(position, 14, 0);
        $.doc_string = reader.string_(position, 16, null);
        $.graph = reader.table(position, 18, onnxruntime.fbs.Graph.decode);
        $.graph_doc_string = reader.string_(position, 20, null);
        $.metadata_props = reader.tableArray(position, 22, onnxruntime.fbs.StringStringEntry.decode);
        return $;
    }
};

onnxruntime.fbs.DeprecatedKernelCreateInfos = class DeprecatedKernelCreateInfos {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.DeprecatedKernelCreateInfos();
        $.node_indices = reader.typedArray(position, 4, Uint32Array);
        $.kernel_def_hashes = reader.uint64s_(position, 6);
        return $;
    }
};

onnxruntime.fbs.DeprecatedSubGraphSessionState = class DeprecatedSubGraphSessionState {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.DeprecatedSubGraphSessionState();
        $.graph_id = reader.string_(position, 4, null);
        $.session_state = reader.table(position, 6, onnxruntime.fbs.DeprecatedSessionState.decode);
        return $;
    }
};

onnxruntime.fbs.DeprecatedSessionState = class DeprecatedSessionState {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.DeprecatedSessionState();
        $.kernels = reader.table(position, 4, onnxruntime.fbs.DeprecatedKernelCreateInfos.decode);
        $.sub_graph_session_states = reader.tableArray(position, 6, onnxruntime.fbs.DeprecatedSubGraphSessionState.decode);
        return $;
    }
};

onnxruntime.fbs.ArgType = {
    INPUT: 0,
    OUTPUT: 1
};

onnxruntime.fbs.ArgTypeAndIndex = class ArgTypeAndIndex {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.ArgTypeAndIndex();
        $.arg_type = reader.int8_(position, 4, 0);
        $.index = reader.uint32_(position, 6, 0);
        return $;
    }
};

onnxruntime.fbs.KernelTypeStrArgsEntry = class KernelTypeStrArgsEntry {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.KernelTypeStrArgsEntry();
        $.kernel_type_str = reader.string_(position, 4, null);
        $.args = reader.tableArray(position, 6, onnxruntime.fbs.ArgTypeAndIndex.decode);
        return $;
    }
};

onnxruntime.fbs.OpIdKernelTypeStrArgsEntry = class OpIdKernelTypeStrArgsEntry {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.OpIdKernelTypeStrArgsEntry();
        $.op_id = reader.string_(position, 4, null);
        $.kernel_type_str_args = reader.tableArray(position, 6, onnxruntime.fbs.KernelTypeStrArgsEntry.decode);
        return $;
    }
};

onnxruntime.fbs.KernelTypeStrResolver = class KernelTypeStrResolver {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.KernelTypeStrResolver();
        $.op_kernel_type_str_args = reader.tableArray(position, 4, onnxruntime.fbs.OpIdKernelTypeStrArgsEntry.decode);
        return $;
    }
};

onnxruntime.fbs.InferenceSession = class InferenceSession {

    static identifier(reader) {
        return reader.identifier === 'ORTM';
    }

    static create(reader) {
        return onnxruntime.fbs.InferenceSession.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.InferenceSession();
        $.ort_version = reader.string_(position, 4, null);
        $.model = reader.table(position, 6, onnxruntime.fbs.Model.decode);
        $.session_state = reader.table(position, 8, onnxruntime.fbs.DeprecatedSessionState.decode);
        $.kernel_type_str_resolver = reader.table(position, 10, onnxruntime.fbs.KernelTypeStrResolver.decode);
        return $;
    }
};
