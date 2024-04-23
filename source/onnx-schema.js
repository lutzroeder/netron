
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
        $.dim = reader.tables(position, 4, onnxruntime.fbs.Dimension);
        return $;
    }
};

onnxruntime.fbs.Dimension = class Dimension {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.Dimension();
        $.value = reader.table(position, 4, onnxruntime.fbs.DimensionValue);
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
        $.dim_value = reader.int64_(position, 6, 0n);
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
        $.shape = reader.table(position, 6, onnxruntime.fbs.Shape);
        return $;
    }
};

onnxruntime.fbs.MapType = class MapType {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.MapType();
        $.key_type = reader.int32_(position, 4, 0);
        $.value_type = reader.table(position, 6, onnxruntime.fbs.TypeInfo);
        return $;
    }
};

onnxruntime.fbs.SequenceType = class SequenceType {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.SequenceType();
        $.elem_type = reader.table(position, 4, onnxruntime.fbs.TypeInfo);
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
        $.input_edges = reader.structs(position, 6, onnxruntime.fbs.EdgeEnd);
        $.output_edges = reader.structs(position, 8, onnxruntime.fbs.EdgeEnd);
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
        $.attributes = reader.tables(position, 24, onnxruntime.fbs.Attribute);
        $.input_arg_counts = reader.array(position, 26, Int32Array);
        $.implicit_inputs = reader.strings_(position, 28);
        return $;
    }
};

onnxruntime.fbs.ValueInfo = class ValueInfo {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.ValueInfo();
        $.name = reader.string_(position, 4, null);
        $.doc_string = reader.string_(position, 6, null);
        $.type = reader.table(position, 8, onnxruntime.fbs.TypeInfo);
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
        $.value = reader.union(position, 6, onnxruntime.fbs.TypeInfoValue);
        return $;
    }
};

onnxruntime.fbs.OperatorSetId = class OperatorSetId {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.OperatorSetId();
        $.domain = reader.string_(position, 4, null);
        $.version = reader.int64_(position, 6, 0n);
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
        $.raw_data = reader.array(position, 12, Uint8Array);
        $.string_data = reader.strings_(position, 14);
        $.external_data_offset = reader.int64_(position, 16, -1n);
        return $;
    }
};

onnxruntime.fbs.SparseTensor = class SparseTensor {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.SparseTensor();
        $.values = reader.table(position, 4, onnxruntime.fbs.Tensor);
        $.indices = reader.table(position, 6, onnxruntime.fbs.Tensor);
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
        $.i = reader.int64_(position, 12, 0n);
        $.s = reader.string_(position, 14, null);
        $.t = reader.table(position, 16, onnxruntime.fbs.Tensor);
        $.g = reader.table(position, 18, onnxruntime.fbs.Graph);
        $.floats = reader.array(position, 20, Float32Array);
        $.ints = reader.int64s_(position, 22);
        $.strings = reader.strings_(position, 24);
        $.tensors = reader.tables(position, 26, onnxruntime.fbs.Tensor);
        $.graphs = reader.tables(position, 28, onnxruntime.fbs.Graph);
        return $;
    }
};

onnxruntime.fbs.NodesToOptimizeIndices = class NodesToOptimizeIndices {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.NodesToOptimizeIndices();
        $.node_indices = reader.array(position, 4, Uint32Array);
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
        $.kernel_def_hash = reader.uint64_(position, 6, 0n);
        return $;
    }
};

onnxruntime.fbs.RuntimeOptimizationRecord = class RuntimeOptimizationRecord {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.RuntimeOptimizationRecord();
        $.action_id = reader.string_(position, 4, null);
        $.nodes_to_optimize_indices = reader.table(position, 6, onnxruntime.fbs.NodesToOptimizeIndices);
        $.produced_nodes = reader.tables(position, 8, onnxruntime.fbs.DeprecatedNodeIndexAndKernelDefHash);
        $.produced_op_ids = reader.strings_(position, 10);
        return $;
    }
};

onnxruntime.fbs.RuntimeOptimizationRecordContainerEntry = class RuntimeOptimizationRecordContainerEntry {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.RuntimeOptimizationRecordContainerEntry();
        $.optimizer_name = reader.string_(position, 4, null);
        $.runtime_optimization_records = reader.tables(position, 6, onnxruntime.fbs.RuntimeOptimizationRecord);
        return $;
    }
};

onnxruntime.fbs.RuntimeOptimizations = class RuntimeOptimizations {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.RuntimeOptimizations();
        $.records = reader.tables(position, 4, onnxruntime.fbs.RuntimeOptimizationRecordContainerEntry);
        return $;
    }
};

onnxruntime.fbs.Graph = class Graph {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.Graph();
        $.initializers = reader.tables(position, 4, onnxruntime.fbs.Tensor);
        $.node_args = reader.tables(position, 6, onnxruntime.fbs.ValueInfo);
        $.nodes = reader.tables(position, 8, onnxruntime.fbs.Node);
        $.max_node_index = reader.uint32_(position, 10, 0);
        $.node_edges = reader.tables(position, 12, onnxruntime.fbs.NodeEdge);
        $.inputs = reader.strings_(position, 14);
        $.outputs = reader.strings_(position, 16);
        $.sparse_initializers = reader.tables(position, 18, onnxruntime.fbs.SparseTensor);
        $.runtime_optimizations = reader.table(position, 20, onnxruntime.fbs.RuntimeOptimizations);
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
        $.ir_version = reader.int64_(position, 4, 0n);
        $.opset_import = reader.tables(position, 6, onnxruntime.fbs.OperatorSetId);
        $.producer_name = reader.string_(position, 8, null);
        $.producer_version = reader.string_(position, 10, null);
        $.domain = reader.string_(position, 12, null);
        $.model_version = reader.int64_(position, 14, 0n);
        $.doc_string = reader.string_(position, 16, null);
        $.graph = reader.table(position, 18, onnxruntime.fbs.Graph);
        $.graph_doc_string = reader.string_(position, 20, null);
        $.metadata_props = reader.tables(position, 22, onnxruntime.fbs.StringStringEntry);
        return $;
    }
};

onnxruntime.fbs.DeprecatedKernelCreateInfos = class DeprecatedKernelCreateInfos {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.DeprecatedKernelCreateInfos();
        $.node_indices = reader.array(position, 4, Uint32Array);
        $.kernel_def_hashes = reader.uint64s_(position, 6);
        return $;
    }
};

onnxruntime.fbs.DeprecatedSubGraphSessionState = class DeprecatedSubGraphSessionState {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.DeprecatedSubGraphSessionState();
        $.graph_id = reader.string_(position, 4, null);
        $.session_state = reader.table(position, 6, onnxruntime.fbs.DeprecatedSessionState);
        return $;
    }
};

onnxruntime.fbs.DeprecatedSessionState = class DeprecatedSessionState {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.DeprecatedSessionState();
        $.kernels = reader.table(position, 4, onnxruntime.fbs.DeprecatedKernelCreateInfos);
        $.sub_graph_session_states = reader.tables(position, 6, onnxruntime.fbs.DeprecatedSubGraphSessionState);
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
        $.args = reader.tables(position, 6, onnxruntime.fbs.ArgTypeAndIndex);
        return $;
    }
};

onnxruntime.fbs.OpIdKernelTypeStrArgsEntry = class OpIdKernelTypeStrArgsEntry {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.OpIdKernelTypeStrArgsEntry();
        $.op_id = reader.string_(position, 4, null);
        $.kernel_type_str_args = reader.tables(position, 6, onnxruntime.fbs.KernelTypeStrArgsEntry);
        return $;
    }
};

onnxruntime.fbs.KernelTypeStrResolver = class KernelTypeStrResolver {

    static decode(reader, position) {
        const $ = new onnxruntime.fbs.KernelTypeStrResolver();
        $.op_kernel_type_str_args = reader.tables(position, 4, onnxruntime.fbs.OpIdKernelTypeStrArgsEntry);
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
        $.model = reader.table(position, 6, onnxruntime.fbs.Model);
        $.session_state = reader.table(position, 8, onnxruntime.fbs.DeprecatedSessionState);
        $.kernel_type_str_resolver = reader.table(position, 10, onnxruntime.fbs.KernelTypeStrResolver);
        return $;
    }
};
