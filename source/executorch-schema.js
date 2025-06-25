
export const executorch_flatbuffer = {};

export const fb_xnnpack = {};

export const vkgraph = {};

executorch_flatbuffer.ScalarType = {
    BYTE: 0,
    CHAR: 1,
    SHORT: 2,
    INT: 3,
    LONG: 4,
    HALF: 5,
    FLOAT: 6,
    DOUBLE: 7,
    BOOL: 11,
    QINT8: 12,
    QUINT8: 13,
    QINT32: 14,
    QUINT4X2: 16,
    QUINT2X4: 17,
    BITS16: 22,
    FLOAT8E5M2: 23,
    FLOAT8E4M3FN: 24,
    FLOAT8E5M2FNUZ: 25,
    FLOAT8E4M3FNUZ: 26,
    UINT16: 27,
    UINT32: 28,
    UINT64: 29
};

executorch_flatbuffer.ContainerMetadata = class ContainerMetadata {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.ContainerMetadata();
        $.encoded_inp_str = reader.string_(position, 4, null);
        $.encoded_out_str = reader.string_(position, 6, null);
        return $;
    }
};

executorch_flatbuffer.Null = class Null {

    static decode(/* reader, position */) {
        const $ = new executorch_flatbuffer.Null();
        return $;
    }
};

executorch_flatbuffer.AllocationDetails = class AllocationDetails {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.AllocationDetails();
        $.memory_id = reader.uint32_(position, 4, 0);
        $.memory_offset_low = reader.uint32_(position, 6, 0);
        $.memory_offset_high = reader.uint32_(position, 8, 0);
        return $;
    }
};

executorch_flatbuffer.TensorShapeDynamism = {
    STATIC: 0,
    DYNAMIC_BOUND: 1,
    DYNAMIC_UNBOUND: 2
};

executorch_flatbuffer.TensorDataLocation = {
    SEGMENT: 0,
    EXTERNAL: 1
};

executorch_flatbuffer.ExtraTensorInfo = class ExtraTensorInfo {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.ExtraTensorInfo();
        $.mutable_data_segments_idx = reader.uint64_(position, 4, 0n);
        $.fully_qualified_name = reader.string_(position, 6, null);
        $.location = reader.int8_(position, 8, 0);
        return $;
    }
};

executorch_flatbuffer.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.Tensor();
        $.scalar_type = reader.int8_(position, 4, 0);
        $.storage_offset = reader.int32_(position, 6, 0);
        $.sizes = reader.array(position, 8, Int32Array);
        $.dim_order = reader.array(position, 10, Uint8Array);
        $.requires_grad = reader.bool_(position, 12, false);
        $.data_buffer_idx = reader.uint32_(position, 14, 0);
        $.allocation_info = reader.table(position, 16, executorch_flatbuffer.AllocationDetails);
        $.layout = reader.int8_(position, 18, 0);
        $.shape_dynamism = reader.int8_(position, 20, 0);
        $.extra_tensor_info = reader.table(position, 22, executorch_flatbuffer.ExtraTensorInfo);
        return $;
    }
};

executorch_flatbuffer.Int = class Int {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.Int();
        $.int_val = reader.int64_(position, 4, 0n);
        return $;
    }
};

executorch_flatbuffer.Bool = class Bool {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.Bool();
        $.bool_val = reader.bool_(position, 4, false);
        return $;
    }
};

executorch_flatbuffer.Double = class Double {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.Double();
        $.double_val = reader.float64_(position, 4, 0);
        return $;
    }
};

executorch_flatbuffer.String = class String {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.String();
        $.string_val = reader.string_(position, 4, null);
        return $;
    }
};

executorch_flatbuffer.IntList = class IntList {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.IntList();
        $.items = reader.int64s_(position, 4);
        return $;
    }
};

executorch_flatbuffer.DoubleList = class DoubleList {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.DoubleList();
        $.items = reader.array(position, 4, Float64Array);
        return $;
    }
};

executorch_flatbuffer.BoolList = class BoolList {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.BoolList();
        $.items = reader.bools_(position, 4);
        return $;
    }
};

executorch_flatbuffer.TensorList = class TensorList {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.TensorList();
        $.items = reader.array(position, 4, Int32Array);
        return $;
    }
};

executorch_flatbuffer.OptionalTensorList = class OptionalTensorList {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.OptionalTensorList();
        $.items = reader.array(position, 4, Int32Array);
        return $;
    }
};

executorch_flatbuffer.KernelTypes = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return executorch_flatbuffer.Null.decode(reader, position);
            case 2: return executorch_flatbuffer.Int.decode(reader, position);
            case 3: return executorch_flatbuffer.Bool.decode(reader, position);
            case 4: return executorch_flatbuffer.Double.decode(reader, position);
            case 5: return executorch_flatbuffer.Tensor.decode(reader, position);
            case 6: return executorch_flatbuffer.String.decode(reader, position);
            case 7: return executorch_flatbuffer.IntList.decode(reader, position);
            case 8: return executorch_flatbuffer.DoubleList.decode(reader, position);
            case 9: return executorch_flatbuffer.BoolList.decode(reader, position);
            case 10: return executorch_flatbuffer.TensorList.decode(reader, position);
            case 11: return executorch_flatbuffer.OptionalTensorList.decode(reader, position);
            default: return undefined;
        }
    }
};

executorch_flatbuffer.EValue = class EValue {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.EValue();
        $.val = reader.union(position, 4, executorch_flatbuffer.KernelTypes);
        return $;
    }
};

executorch_flatbuffer.Operator = class Operator {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.Operator();
        $.name = reader.string_(position, 4, null);
        $.overload = reader.string_(position, 6, null);
        return $;
    }
};

executorch_flatbuffer.KernelCall = class KernelCall {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.KernelCall();
        $.op_index = reader.int32_(position, 4, 0);
        $.args = reader.array(position, 6, Int32Array);
        return $;
    }
};

executorch_flatbuffer.DelegateCall = class DelegateCall {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.DelegateCall();
        $.delegate_index = reader.int32_(position, 4, 0);
        $.args = reader.array(position, 6, Int32Array);
        return $;
    }
};

executorch_flatbuffer.MoveCall = class MoveCall {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.MoveCall();
        $.move_from = reader.int32_(position, 4, 0);
        $.move_to = reader.int32_(position, 6, 0);
        return $;
    }
};

executorch_flatbuffer.JumpFalseCall = class JumpFalseCall {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.JumpFalseCall();
        $.cond_value_index = reader.int32_(position, 4, 0);
        $.destination_instruction = reader.int32_(position, 6, 0);
        return $;
    }
};

executorch_flatbuffer.FreeCall = class FreeCall {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.FreeCall();
        $.value_index = reader.int32_(position, 4, 0);
        return $;
    }
};

executorch_flatbuffer.InstructionArguments = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return executorch_flatbuffer.KernelCall.decode(reader, position);
            case 2: return executorch_flatbuffer.DelegateCall.decode(reader, position);
            case 3: return executorch_flatbuffer.MoveCall.decode(reader, position);
            case 4: return executorch_flatbuffer.JumpFalseCall.decode(reader, position);
            case 5: return executorch_flatbuffer.FreeCall.decode(reader, position);
            default: return undefined;
        }
    }
};

executorch_flatbuffer.Instruction = class Instruction {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.Instruction();
        $.instr_args = reader.union(position, 4, executorch_flatbuffer.InstructionArguments);
        return $;
    }
};

executorch_flatbuffer.Frame = class Frame {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.Frame();
        $.filename = reader.string_(position, 4, null);
        $.lineno = reader.int32_(position, 6, 0);
        $.name = reader.string_(position, 8, null);
        $.context = reader.string_(position, 10, null);
        return $;
    }
};

executorch_flatbuffer.FrameList = class FrameList {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.FrameList();
        $.items = reader.tables(position, 4, executorch_flatbuffer.Frame);
        return $;
    }
};

executorch_flatbuffer.DataLocation = {
    INLINE: 0,
    SEGMENT: 1
};

executorch_flatbuffer.BackendDelegateDataReference = class BackendDelegateDataReference {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.BackendDelegateDataReference();
        $.location = reader.int8_(position, 4, 0);
        $.index = reader.uint32_(position, 6, 0);
        return $;
    }
};

executorch_flatbuffer.CompileSpec = class CompileSpec {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.CompileSpec();
        $.key = reader.string_(position, 4, null);
        $.value = reader.array(position, 6, Uint8Array);
        return $;
    }
};

executorch_flatbuffer.BackendDelegate = class BackendDelegate {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.BackendDelegate();
        $.id = reader.string_(position, 4, null);
        $.processed = reader.table(position, 6, executorch_flatbuffer.BackendDelegateDataReference);
        $.compile_specs = reader.tables(position, 8, executorch_flatbuffer.CompileSpec);
        return $;
    }
};

executorch_flatbuffer.Chain = class Chain {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.Chain();
        $.inputs = reader.array(position, 4, Int32Array);
        $.outputs = reader.array(position, 6, Int32Array);
        $.instructions = reader.tables(position, 8, executorch_flatbuffer.Instruction);
        $.stacktrace = reader.tables(position, 10, executorch_flatbuffer.FrameList);
        return $;
    }
};

executorch_flatbuffer.ExecutionPlan = class ExecutionPlan {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.ExecutionPlan();
        $.name = reader.string_(position, 4, null);
        $.container_meta_type = reader.table(position, 6, executorch_flatbuffer.ContainerMetadata);
        $.values = reader.tables(position, 8, executorch_flatbuffer.EValue);
        $.inputs = reader.array(position, 10, Int32Array);
        $.outputs = reader.array(position, 12, Int32Array);
        $.chains = reader.tables(position, 14, executorch_flatbuffer.Chain);
        $.operators = reader.tables(position, 16, executorch_flatbuffer.Operator);
        $.delegates = reader.tables(position, 18, executorch_flatbuffer.BackendDelegate);
        $.non_const_buffer_sizes = reader.int64s_(position, 20);
        return $;
    }
};

executorch_flatbuffer.Buffer = class Buffer {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.Buffer();
        $.storage = reader.array(position, 4, Uint8Array);
        return $;
    }
};

executorch_flatbuffer.BackendDelegateInlineData = class BackendDelegateInlineData {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.BackendDelegateInlineData();
        $.data = reader.array(position, 4, Uint8Array);
        return $;
    }
};

executorch_flatbuffer.DataSegment = class DataSegment {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.DataSegment();
        $.offset = reader.uint64_(position, 4, 0n);
        $.size = reader.uint64_(position, 6, 0n);
        return $;
    }
};

executorch_flatbuffer.SubsegmentOffsets = class SubsegmentOffsets {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.SubsegmentOffsets();
        $.segment_index = reader.uint32_(position, 4, 0);
        $.offsets = reader.uint64s_(position, 6);
        return $;
    }
};

executorch_flatbuffer.NamedData = class NamedData {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.NamedData();
        $.key = reader.string_(position, 4, null);
        $.segment_index = reader.uint32_(position, 6, 0);
        return $;
    }
};

executorch_flatbuffer.Program = class Program {

    static identifier(reader) {
        return reader.identifier === 'ET12';
    }

    static create(reader) {
        return executorch_flatbuffer.Program.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.Program();
        $.version = reader.uint32_(position, 4, 0);
        $.execution_plan = reader.tables(position, 6, executorch_flatbuffer.ExecutionPlan);
        $.constant_buffer = reader.tables(position, 8, executorch_flatbuffer.Buffer);
        $.backend_delegate_data = reader.tables(position, 10, executorch_flatbuffer.BackendDelegateInlineData);
        $.segments = reader.tables(position, 12, executorch_flatbuffer.DataSegment);
        $.constant_segment = reader.table(position, 14, executorch_flatbuffer.SubsegmentOffsets);
        $.mutable_data_segments = reader.tables(position, 16, executorch_flatbuffer.SubsegmentOffsets);
        $.named_data = reader.tables(position, 18, executorch_flatbuffer.NamedData);
        return $;
    }
};

fb_xnnpack.XNNDatatype = {
    xnn_datatype_invalid: 0,
    xnn_datatype_fp32: 1,
    xnn_datatype_fp16: 2,
    xnn_datatype_qint8: 3,
    xnn_datatype_quint8: 4,
    xnn_datatype_qint32: 5,
    xnn_datatype_qcint8: 6,
    xnn_datatype_qcint32: 7,
    xnn_datatype_qcint4: 8,
    xnn_datatype_qdint8: 9,
    xnn_datatype_qbint4: 10,
    xnn_datatype_qpint8: 11,
    xnn_datatype_int32: 12,
    xnn_datatype_pfp32: 13,
    xnn_datatype_bf16: 14
};

fb_xnnpack.XNNQuantParams = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return fb_xnnpack.PerChannelQuant.decode(reader, position);
            case 2: return fb_xnnpack.PerTensorQuant.decode(reader, position);
            case 3: return fb_xnnpack.PerTokenDynamicQuant.decode(reader, position);
            case 4: return fb_xnnpack.PerChannelGroupQuant.decode(reader, position);
            default: return undefined;
        }
    }
};

fb_xnnpack.Buffer = class Buffer {

    static decode(reader, position) {
        const $ = new fb_xnnpack.Buffer();
        $.storage = reader.array(position, 4, Uint8Array);
        return $;
    }
};

fb_xnnpack.PerChannelGroupQuant = class PerChannelGroupQuant {

    static decode(reader, position) {
        const $ = new fb_xnnpack.PerChannelGroupQuant();
        $.scale = reader.array(position, 4, Float32Array);
        $.channel_dim = reader.int32_(position, 6, 0);
        $.group_size = reader.int32_(position, 8, 0);
        $.scale_bf16 = reader.array(position, 10, Uint16Array);
        $.scale_buffer_idx = reader.uint32_(position, 12, 0);
        $.num_scales = reader.uint32_(position, 14, 0);
        return $;
    }
};

fb_xnnpack.PerChannelQuant = class PerChannelQuant {

    static decode(reader, position) {
        const $ = new fb_xnnpack.PerChannelQuant();
        $.scale = reader.array(position, 4, Float32Array);
        $.channel_dim = reader.int32_(position, 6, 0);
        $.scale_buffer_idx = reader.uint32_(position, 8, 0);
        $.num_scales = reader.uint32_(position, 10, 0);
        return $;
    }
};

fb_xnnpack.PerTokenDynamicQuant = class PerTokenDynamicQuant {

    static decode(reader, position) {
        const $ = new fb_xnnpack.PerTokenDynamicQuant();
        $.num_nonbatch_dims = reader.int32_(position, 4, 0);
        return $;
    }
};

fb_xnnpack.PerTensorQuant = class PerTensorQuant {

    static decode(reader, position) {
        const $ = new fb_xnnpack.PerTensorQuant();
        $.scale = reader.float32_(position, 4, 0);
        $.zero_point = reader.int32_(position, 6, 0);
        return $;
    }
};

fb_xnnpack.XNNTensorValue = class XNNTensorValue {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNTensorValue();
        $.datatype = reader.int16_(position, 4, 0);
        $.num_dims = reader.uint32_(position, 6, 0);
        $.dims = reader.array(position, 8, Uint32Array);
        $.constant_buffer_idx = reader.uint32_(position, 10, 0);
        $.external_id = reader.uint32_(position, 12, 0);
        $.flags = reader.uint32_(position, 14, 0);
        $.id_out = reader.uint32_(position, 16, 0);
        return $;
    }
};

fb_xnnpack.XNNQuantizedTensorValue = class XNNQuantizedTensorValue {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNQuantizedTensorValue();
        $.tensor_value = reader.table(position, 4, fb_xnnpack.XNNTensorValue);
        $.quant_params = reader.union(position, 6, fb_xnnpack.XNNQuantParams);
        return $;
    }
};

fb_xnnpack.XNodeUnion = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return fb_xnnpack.XNNAdd.decode(reader, position);
            case 2: return fb_xnnpack.XNNFullyConnected.decode(reader, position);
            case 3: return fb_xnnpack.XNNSoftmax.decode(reader, position);
            case 4: return fb_xnnpack.XNNSigmoid.decode(reader, position);
            case 5: return fb_xnnpack.XNNStaticTranspose.decode(reader, position);
            case 6: return fb_xnnpack.XNNClamp.decode(reader, position);
            case 7: return fb_xnnpack.XNNConv2d.decode(reader, position);
            case 8: return fb_xnnpack.XNNDiv.decode(reader, position);
            case 9: return fb_xnnpack.XNNStaticResizeBilinear2D.decode(reader, position);
            case 10: return fb_xnnpack.XNNStaticConstantPad.decode(reader, position);
            case 11: return fb_xnnpack.XNNAvgPooling2d.decode(reader, position);
            case 12: return fb_xnnpack.XNNMinimum.decode(reader, position);
            case 13: return fb_xnnpack.XNNDepthwiseConv2d.decode(reader, position);
            case 14: return fb_xnnpack.XNNMaxPooling2d.decode(reader, position);
            case 15: return fb_xnnpack.XNNMultiply.decode(reader, position);
            case 16: return fb_xnnpack.XNNSubtract.decode(reader, position);
            case 17: return fb_xnnpack.XNNFloor.decode(reader, position);
            case 18: return fb_xnnpack.XNNConvert.decode(reader, position);
            case 19: return fb_xnnpack.XNNGlobalAvgPooling2d.decode(reader, position);
            case 20: return fb_xnnpack.XNNStaticReshape.decode(reader, position);
            case 21: return fb_xnnpack.XNNArgMaxPooling2d.decode(reader, position);
            case 22: return fb_xnnpack.XNNSquareRoot.decode(reader, position);
            case 23: return fb_xnnpack.XNNCeiling.decode(reader, position);
            case 24: return fb_xnnpack.XNNHardswish.decode(reader, position);
            case 25: return fb_xnnpack.XNNLeakyReLU.decode(reader, position);
            case 26: return fb_xnnpack.XNNMaximum.decode(reader, position);
            case 27: return fb_xnnpack.XNNNegate.decode(reader, position);
            case 28: return fb_xnnpack.XNNSquare.decode(reader, position);
            case 29: return fb_xnnpack.XNNELU.decode(reader, position);
            case 30: return fb_xnnpack.XNNAbs.decode(reader, position);
            case 31: return fb_xnnpack.XNNPReLU.decode(reader, position);
            case 32: return fb_xnnpack.XNNConcatenate2.decode(reader, position);
            case 33: return fb_xnnpack.XNNConcatenate3.decode(reader, position);
            case 34: return fb_xnnpack.XNNConcatenate4.decode(reader, position);
            case 35: return fb_xnnpack.XNNStaticSlice.decode(reader, position);
            case 36: return fb_xnnpack.XNNScaledDotProductAttention.decode(reader, position);
            case 37: return fb_xnnpack.XNNBatchMatrixMultiply.decode(reader, position);
            case 38: return fb_xnnpack.XNNConcatenate5.decode(reader, position);
            case 39: return fb_xnnpack.XNNConvTranspose2d.decode(reader, position);
            case 40: return fb_xnnpack.XNNReciprocalSquareRoot.decode(reader, position);
            case 41: return fb_xnnpack.XNNLog.decode(reader, position);
            case 42: return fb_xnnpack.XNNGelu.decode(reader, position);
            case 43: return fb_xnnpack.XNNTanh.decode(reader, position);
            case 44: return fb_xnnpack.XNNExp.decode(reader, position);
            default: return undefined;
        }
    }
};

fb_xnnpack.XValueUnion = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return fb_xnnpack.XNNTensorValue.decode(reader, position);
            case 2: return fb_xnnpack.XNNQuantizedTensorValue.decode(reader, position);
            default: return undefined;
        }
    }
};

fb_xnnpack.OutputMinMax = class OutputMinMax {

    static decode(reader, position) {
        const $ = new fb_xnnpack.OutputMinMax();
        $.output_min = reader.float32_(position, 4, 0);
        $.output_max = reader.float32_(position, 6, 0);
        return $;
    }
};

fb_xnnpack.XNode = class XNode {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNode();
        $.xnode_union = reader.union(position, 4, fb_xnnpack.XNodeUnion);
        $.debug_handle = reader.uint32_(position, 8, 0);
        $.output_min_max = reader.table(position, 10, fb_xnnpack.OutputMinMax);
        return $;
    }
};

fb_xnnpack.XValue = class XValue {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XValue();
        $.xvalue_union = reader.union(position, 4, fb_xnnpack.XValueUnion);
        return $;
    }
};

fb_xnnpack.XNNStaticTranspose = class XNNStaticTranspose {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNStaticTranspose();
        $.num_dims = reader.uint32_(position, 4, 0);
        $.perm = reader.array(position, 6, Uint32Array);
        $.input_id = reader.uint32_(position, 8, 0);
        $.output_id = reader.uint32_(position, 10, 0);
        $.flags = reader.uint32_(position, 12, 0);
        return $;
    }
};

fb_xnnpack.XNNStaticResizeBilinear2D = class XNNStaticResizeBilinear2D {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNStaticResizeBilinear2D();
        $.new_height = reader.uint32_(position, 4, 0);
        $.new_width = reader.uint32_(position, 6, 0);
        $.input_id = reader.uint32_(position, 8, 0);
        $.output_id = reader.uint32_(position, 10, 0);
        $.flags = reader.uint32_(position, 12, 0);
        return $;
    }
};

fb_xnnpack.XNNStaticConstantPad = class XNNStaticConstantPad {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNStaticConstantPad();
        $.pre_paddings = reader.array(position, 4, Uint32Array);
        $.post_paddings = reader.array(position, 6, Uint32Array);
        $.padding_value = reader.float32_(position, 8, 0);
        $.input_id = reader.uint32_(position, 10, 0);
        $.output_id = reader.uint32_(position, 12, 0);
        $.flags = reader.uint32_(position, 14, 0);
        return $;
    }
};

fb_xnnpack._XNNNode2x1 = class _XNNNode2x1 {

    static decode(reader, position, $) {
        $ = $ || new fb_xnnpack._XNNNode2x1();
        $.input1_id = reader.uint32_(position, 4, 0);
        $.input2_id = reader.uint32_(position, 6, 0);
        $.output_id = reader.uint32_(position, 8, 0);
        $.flags = reader.uint32_(position, 10, 0);
        return $;
    }
};

fb_xnnpack._XNNNode1x1 = class _XNNNode1x1 {

    static decode(reader, position, $) {
        $ = $ || new fb_xnnpack._XNNNode1x1();
        $.input_id = reader.uint32_(position, 4, 0);
        $.output_id = reader.uint32_(position, 6, 0);
        $.flags = reader.uint32_(position, 8, 0);
        return $;
    }
};

fb_xnnpack._XNNCat = class _XNNCat {

    static decode(reader, position, $) {
        $ = $ || new fb_xnnpack._XNNCat();
        $.axis = reader.uint32_(position, 4, 0);
        $.input1_id = reader.uint32_(position, 6, 0);
        $.input2_id = reader.uint32_(position, 8, 0);
        $.input3_id = reader.uint32_(position, 10, 0);
        $.input4_id = reader.uint32_(position, 12, 0);
        $.output_id = reader.uint32_(position, 14, 0);
        $.flags = reader.uint32_(position, 16, 0);
        $.input5_id = reader.uint32_(position, 18, 0);
        return $;
    }
};

fb_xnnpack.XNNELU = class XNNELU {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNELU();
        $.alpha = reader.float32_(position, 4, 0);
        $.input_id = reader.uint32_(position, 6, 0);
        $.output_id = reader.uint32_(position, 8, 0);
        $.flags = reader.uint32_(position, 10, 0);
        return $;
    }
};

fb_xnnpack.XNNFullyConnected = class XNNFullyConnected {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNFullyConnected();
        $.input1_id = reader.uint32_(position, 4, 0);
        $.filter_id = reader.uint32_(position, 6, 0);
        $.bias_id = reader.uint32_(position, 8, 0);
        $.output_id = reader.uint32_(position, 10, 0);
        $.flags = reader.uint32_(position, 12, 0);
        return $;
    }
};

fb_xnnpack._XNNNodeConv = class _XNNNodeConv {

    static decode(reader, position, $) {
        $ = $ || new fb_xnnpack._XNNNodeConv();
        $.padding_top = reader.uint32_(position, 4, 0);
        $.padding_right = reader.uint32_(position, 6, 0);
        $.padding_bottom = reader.uint32_(position, 8, 0);
        $.padding_left = reader.uint32_(position, 10, 0);
        $.kernel_height = reader.uint32_(position, 12, 0);
        $.kernel_width = reader.uint32_(position, 14, 0);
        $.subsampling_height = reader.uint32_(position, 16, 0);
        $.subsampling_width = reader.uint32_(position, 18, 0);
        $.dilation_height = reader.uint32_(position, 20, 0);
        $.dilation_width = reader.uint32_(position, 22, 0);
        $.group_input_channels = reader.uint32_(position, 24, 0);
        $.group_output_channels = reader.uint32_(position, 26, 0);
        $.groups = reader.uint32_(position, 28, 0);
        $.adjustment_height = reader.uint32_(position, 30, 0);
        $.adjustment_width = reader.uint32_(position, 32, 0);
        $.input1_id = reader.uint32_(position, 34, 0);
        $.filter_id = reader.uint32_(position, 36, 0);
        $.bias_id = reader.uint32_(position, 38, 0);
        $.output_id = reader.uint32_(position, 40, 0);
        $.flags = reader.uint32_(position, 42, 0);
        return $;
    }
};

fb_xnnpack._XNNPooling2D = class _XNNPooling2D {

    static decode(reader, position, $) {
        $ = $ || new fb_xnnpack._XNNPooling2D();
        $.padding_top = reader.uint32_(position, 4, 0);
        $.padding_right = reader.uint32_(position, 6, 0);
        $.padding_bottom = reader.uint32_(position, 8, 0);
        $.padding_left = reader.uint32_(position, 10, 0);
        $.pooling_height = reader.uint32_(position, 12, 0);
        $.pooling_width = reader.uint32_(position, 14, 0);
        $.stride_height = reader.uint32_(position, 16, 0);
        $.stride_width = reader.uint32_(position, 18, 0);
        $.dilation_height = reader.uint32_(position, 20, 0);
        $.dilation_width = reader.uint32_(position, 22, 0);
        $.input_id = reader.uint32_(position, 24, 0);
        $.output_id = reader.uint32_(position, 26, 0);
        $.flags = reader.uint32_(position, 28, 0);
        return $;
    }
};

fb_xnnpack.XNNStaticReshape = class XNNStaticReshape {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNStaticReshape();
        $.num_dims = reader.uint32_(position, 4, 0);
        $.new_shape = reader.array(position, 6, Uint32Array);
        $.input_id = reader.uint32_(position, 8, 0);
        $.output_id = reader.uint32_(position, 10, 0);
        $.flags = reader.uint32_(position, 12, 0);
        return $;
    }
};

fb_xnnpack.XNNStaticSlice = class XNNStaticSlice {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNStaticSlice();
        $.num_dims = reader.uint32_(position, 4, 0);
        $.offsets = reader.array(position, 6, Uint32Array);
        $.sizes = reader.array(position, 8, Uint32Array);
        $.input_id = reader.uint32_(position, 10, 0);
        $.output_id = reader.uint32_(position, 12, 0);
        $.flags = reader.uint32_(position, 14, 0);
        return $;
    }
};

fb_xnnpack.XNNScaledDotProductAttention = class XNNScaledDotProductAttention {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNScaledDotProductAttention();
        $.query_id = reader.uint32_(position, 4, 0);
        $.key_id = reader.uint32_(position, 6, 0);
        $.value_id = reader.uint32_(position, 8, 0);
        $.scale_id = reader.uint32_(position, 10, 0);
        $.mask_id = reader.uint32_(position, 12, 0);
        $.output_id = reader.uint32_(position, 14, 0);
        $.flags = reader.uint32_(position, 16, 0);
        return $;
    }
};

fb_xnnpack.XNNArgMaxPooling2d = class XNNArgMaxPooling2d {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNArgMaxPooling2d();
        $.padding_top = reader.uint32_(position, 4, 0);
        $.padding_right = reader.uint32_(position, 6, 0);
        $.padding_bottom = reader.uint32_(position, 8, 0);
        $.padding_left = reader.uint32_(position, 10, 0);
        $.pooling_height = reader.uint32_(position, 12, 0);
        $.pooling_width = reader.uint32_(position, 14, 0);
        $.input_id = reader.uint32_(position, 16, 0);
        $.output_value_id = reader.uint32_(position, 18, 0);
        $.output_index_id = reader.uint32_(position, 20, 0);
        $.flags = reader.uint32_(position, 22, 0);
        return $;
    }
};

fb_xnnpack.XNNLeakyReLU = class XNNLeakyReLU {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNLeakyReLU();
        $.negative_slope = reader.float32_(position, 4, 0);
        $.input_id = reader.uint32_(position, 6, 0);
        $.output_id = reader.uint32_(position, 8, 0);
        $.flags = reader.uint32_(position, 10, 0);
        return $;
    }
};

fb_xnnpack.ConstantDataOffset = class ConstantDataOffset {

    static decode(reader, position) {
        const $ = new fb_xnnpack.ConstantDataOffset();
        $.offset = reader.uint64_(position, 4, 0n);
        $.size = reader.uint64_(position, 6, 0n);
        $.named_key = reader.string_(position, 8, null);
        return $;
    }
};

fb_xnnpack.XNNGraph = class XNNGraph {

    static identifier(reader) {
        return reader.identifier === 'XN01';
    }

    static create(reader) {
        return fb_xnnpack.XNNGraph.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNGraph();
        $.version = reader.string_(position, 4, null);
        $.xnodes = reader.tables(position, 6, fb_xnnpack.XNode);
        $.xvalues = reader.tables(position, 8, fb_xnnpack.XValue);
        $.num_externs = reader.uint32_(position, 10, 0);
        $.input_ids = reader.array(position, 12, Uint32Array);
        $.output_ids = reader.array(position, 14, Uint32Array);
        $.constant_buffer = reader.tables(position, 16, fb_xnnpack.Buffer);
        $.mem_buffer_sizes = reader.array(position, 18, Uint32Array);
        $.constant_data = reader.tables(position, 20, fb_xnnpack.ConstantDataOffset);
        return $;
    }
};

fb_xnnpack.XNNAdd = class XNNAdd {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNAdd();
        fb_xnnpack._XNNNode2x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNSoftmax = class XNNSoftmax {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNSoftmax();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNSigmoid = class XNNSigmoid {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNSigmoid();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNClamp = class XNNClamp {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNClamp();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNConv2d = class XNNConv2d {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNConv2d();
        fb_xnnpack._XNNNodeConv.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNDiv = class XNNDiv {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNDiv();
        fb_xnnpack._XNNNode2x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNAvgPooling2d = class XNNAvgPooling2d {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNAvgPooling2d();
        fb_xnnpack._XNNPooling2D.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNMinimum = class XNNMinimum {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNMinimum();
        fb_xnnpack._XNNNode2x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNDepthwiseConv2d = class XNNDepthwiseConv2d {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNDepthwiseConv2d();
        fb_xnnpack._XNNNodeConv.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNMaxPooling2d = class XNNMaxPooling2d {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNMaxPooling2d();
        fb_xnnpack._XNNPooling2D.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNMultiply = class XNNMultiply {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNMultiply();
        fb_xnnpack._XNNNode2x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNSubtract = class XNNSubtract {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNSubtract();
        fb_xnnpack._XNNNode2x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNFloor = class XNNFloor {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNFloor();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNConvert = class XNNConvert {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNConvert();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNGlobalAvgPooling2d = class XNNGlobalAvgPooling2d {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNGlobalAvgPooling2d();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNSquareRoot = class XNNSquareRoot {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNSquareRoot();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNCeiling = class XNNCeiling {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNCeiling();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNHardswish = class XNNHardswish {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNHardswish();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNMaximum = class XNNMaximum {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNMaximum();
        fb_xnnpack._XNNNode2x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNNegate = class XNNNegate {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNNegate();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNSquare = class XNNSquare {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNSquare();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNAbs = class XNNAbs {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNAbs();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNPReLU = class XNNPReLU {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNPReLU();
        fb_xnnpack._XNNNode2x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNConcatenate2 = class XNNConcatenate2 {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNConcatenate2();
        fb_xnnpack._XNNCat.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNConcatenate3 = class XNNConcatenate3 {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNConcatenate3();
        fb_xnnpack._XNNCat.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNConcatenate4 = class XNNConcatenate4 {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNConcatenate4();
        fb_xnnpack._XNNCat.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNBatchMatrixMultiply = class XNNBatchMatrixMultiply {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNBatchMatrixMultiply();
        fb_xnnpack._XNNNode2x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNConcatenate5 = class XNNConcatenate5 {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNConcatenate5();
        fb_xnnpack._XNNCat.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNConvTranspose2d = class XNNConvTranspose2d {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNConvTranspose2d();
        fb_xnnpack._XNNNodeConv.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNReciprocalSquareRoot = class XNNReciprocalSquareRoot {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNReciprocalSquareRoot();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNLog = class XNNLog {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNLog();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNGelu = class XNNGelu {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNGelu();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNTanh = class XNNTanh {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNTanh();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

fb_xnnpack.XNNExp = class XNNExp {

    static decode(reader, position) {
        const $ = new fb_xnnpack.XNNExp();
        fb_xnnpack._XNNNode1x1.decode(reader, position, $);
        return $;
    }
};

vkgraph.OperatorCall = class OperatorCall {

    static decode(reader, position) {
        const $ = new vkgraph.OperatorCall();
        $.node_id = reader.uint32_(position, 4, 0);
        $.name = reader.string_(position, 6, null);
        $.args = reader.array(position, 8, Int32Array);
        return $;
    }
};

vkgraph.VkDataType = {
    BOOL: 0,
    UINT8: 1,
    INT8: 2,
    INT32: 3,
    FLOAT16: 4,
    FLOAT32: 5
};

vkgraph.VkStorageType = {
    BUFFER: 0,
    TEXTURE_3D: 1,
    TEXTURE_2D: 2,
    DEFAULT_STORAGE: 255
};

vkgraph.VkMemoryLayout = {
    TENSOR_WIDTH_PACKED: 0,
    TENSOR_HEIGHT_PACKED: 1,
    TENSOR_CHANNELS_PACKED: 2,
    DEFAULT_LAYOUT: 255
};

vkgraph.VkTensor = class VkTensor {

    static decode(reader, position) {
        const $ = new vkgraph.VkTensor();
        $.datatype = reader.int8_(position, 4, 0);
        $.dims = reader.array(position, 6, Uint32Array);
        $.constant_id = reader.int32_(position, 8, 0);
        $.mem_obj_id = reader.int32_(position, 10, 0);
        $.storage_type = reader.uint8_(position, 12, 255);
        $.memory_layout = reader.uint8_(position, 14, 255);
        return $;
    }
};

vkgraph.Null = class Null {

    static decode(/* reader, position */) {
        const $ = new vkgraph.Null();
        return $;
    }
};

vkgraph.Int = class Int {

    static decode(reader, position) {
        const $ = new vkgraph.Int();
        $.int_val = reader.int64_(position, 4, 0n);
        return $;
    }
};

vkgraph.Bool = class Bool {

    static decode(reader, position) {
        const $ = new vkgraph.Bool();
        $.bool_val = reader.bool_(position, 4, false);
        return $;
    }
};

vkgraph.Double = class Double {

    static decode(reader, position) {
        const $ = new vkgraph.Double();
        $.double_val = reader.float64_(position, 4, 0);
        return $;
    }
};

vkgraph.String = class String {

    static decode(reader, position) {
        const $ = new vkgraph.String();
        $.string_val = reader.string_(position, 4, null);
        return $;
    }
};

vkgraph.IntList = class IntList {

    static decode(reader, position) {
        const $ = new vkgraph.IntList();
        $.items = reader.int64s_(position, 4);
        return $;
    }
};

vkgraph.DoubleList = class DoubleList {

    static decode(reader, position) {
        const $ = new vkgraph.DoubleList();
        $.items = reader.array(position, 4, Float64Array);
        return $;
    }
};

vkgraph.BoolList = class BoolList {

    static decode(reader, position) {
        const $ = new vkgraph.BoolList();
        $.items = reader.bools_(position, 4);
        return $;
    }
};

vkgraph.ValueList = class ValueList {

    static decode(reader, position) {
        const $ = new vkgraph.ValueList();
        $.items = reader.array(position, 4, Int32Array);
        return $;
    }
};

vkgraph.SymInt = class SymInt {

    static decode(reader, position) {
        const $ = new vkgraph.SymInt();
        $.value = reader.int32_(position, 4, 0);
        return $;
    }
};

vkgraph.GraphTypes = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return vkgraph.Null.decode(reader, position);
            case 2: return vkgraph.Int.decode(reader, position);
            case 3: return vkgraph.Double.decode(reader, position);
            case 4: return vkgraph.Bool.decode(reader, position);
            case 5: return vkgraph.VkTensor.decode(reader, position);
            case 6: return vkgraph.IntList.decode(reader, position);
            case 7: return vkgraph.DoubleList.decode(reader, position);
            case 8: return vkgraph.BoolList.decode(reader, position);
            case 9: return vkgraph.ValueList.decode(reader, position);
            case 10: return vkgraph.String.decode(reader, position);
            case 11: return vkgraph.SymInt.decode(reader, position);
            default: return undefined;
        }
    }
};

vkgraph.VkValue = class VkValue {

    static decode(reader, position) {
        const $ = new vkgraph.VkValue();
        $.value = reader.union(position, 4, vkgraph.GraphTypes);
        return $;
    }
};

vkgraph.VkBytes = class VkBytes {

    static decode(reader, position) {
        const $ = new vkgraph.VkBytes();
        $.offset = reader.uint64_(position, 4, 0n);
        $.length = reader.uint64_(position, 6, 0n);
        return $;
    }
};

vkgraph.VkGraph = class VkGraph {

    static identifier(reader) {
        return reader.identifier === 'VK00';
    }

    static create(reader) {
        return vkgraph.VkGraph.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new vkgraph.VkGraph();
        $.version = reader.string_(position, 4, null);
        $.chain = reader.tables(position, 6, vkgraph.OperatorCall);
        $.values = reader.tables(position, 8, vkgraph.VkValue);
        $.input_ids = reader.array(position, 10, Uint32Array);
        $.output_ids = reader.array(position, 12, Uint32Array);
        $.constants = reader.tables(position, 14, vkgraph.VkBytes);
        $.shaders = reader.tables(position, 16, vkgraph.VkBytes);
        $.storage_type_override = reader.uint8_(position, 18, 255);
        $.memory_layout_override = reader.uint8_(position, 20, 255);
        return $;
    }
};
