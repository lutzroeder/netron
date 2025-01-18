
export const executorch_flatbuffer = {};

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
        return $;
    }
};
