
export const torch = {};

export const executorch_flatbuffer = {};

torch.jit = torch.jit || {};

torch.jit.mobile = torch.jit.mobile || {};

torch.jit.mobile.serialization = torch.jit.mobile.serialization || {};

torch.jit.mobile.serialization.Int = class Int {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.Int();
        $.int_val = reader.int64(position + 0);
        return $;
    }
};

torch.jit.mobile.serialization.Bool = class Bool {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.Bool();
        $.bool_val = reader.bool(position + 0);
        return $;
    }
};

torch.jit.mobile.serialization.Double = class Double {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.Double();
        $.double_val = reader.float64(position + 0);
        return $;
    }
};

torch.jit.mobile.serialization.PerTensorAffineSchema = class PerTensorAffineSchema {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.PerTensorAffineSchema();
        $.q_scale = reader.float64(position + 0);
        $.q_zero_point = reader.int32(position + 4);
        return $;
    }
};

torch.jit.mobile.serialization.QuantizedSchema = class QuantizedSchema {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.QuantizedSchema();
        $.qscheme = reader.int8_(position, 4, 0);
        $.scale = reader.float64_(position, 6, 0);
        $.zero_point = reader.int32_(position, 8, 0);
        $.scales = reader.table(position, 10, torch.jit.mobile.serialization.TensorMetadata);
        $.zero_points = reader.table(position, 12, torch.jit.mobile.serialization.TensorMetadata);
        $.axis = reader.int32_(position, 14, 0);
        return $;
    }
};

torch.jit.mobile.serialization.TensorMetadata = class TensorMetadata {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.TensorMetadata();
        $.storage_location_index = reader.uint32_(position, 4, 0);
        $.scalar_type = reader.int8_(position, 6, 0);
        $.storage_offset = reader.int32_(position, 8, 0);
        $.sizes = reader.array(position, 10, Int32Array);
        $.strides = reader.array(position, 12, Int32Array);
        $.requires_grad = reader.bool_(position, 14, false);
        $.quantized_schema = reader.table(position, 16, torch.jit.mobile.serialization.QuantizedSchema);
        return $;
    }
};

torch.jit.mobile.serialization.String = class String {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.String();
        $.data = reader.string_(position, 4, null);
        return $;
    }
};

torch.jit.mobile.serialization.Device = class Device {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.Device();
        $.str = reader.string_(position, 4, null);
        return $;
    }
};

torch.jit.mobile.serialization.List = class List {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.List();
        $.items = reader.array(position, 4, Uint32Array);
        $.annotation_str = reader.string_(position, 6, null);
        return $;
    }
};

torch.jit.mobile.serialization.IntList = class IntList {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.IntList();
        $.items = reader.int64s_(position, 4);
        return $;
    }
};

torch.jit.mobile.serialization.DoubleList = class DoubleList {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.DoubleList();
        $.items = reader.array(position, 4, Float64Array);
        return $;
    }
};

torch.jit.mobile.serialization.BoolList = class BoolList {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.BoolList();
        $.items = reader.bools_(position, 4);
        return $;
    }
};

torch.jit.mobile.serialization.Tuple = class Tuple {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.Tuple();
        $.items = reader.array(position, 4, Uint32Array);
        return $;
    }
};

torch.jit.mobile.serialization.Dict = class Dict {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.Dict();
        $.keys = reader.array(position, 4, Uint32Array);
        $.values = reader.array(position, 6, Uint32Array);
        $.annotation_str = reader.string_(position, 8, null);
        return $;
    }
};

torch.jit.mobile.serialization.TypeType = {
    UNSET: 0,
    CLASS_WITH_FIELD: 1,
    CUSTOM_CLASS: 2,
    CLASS_WITH_SETSTATE: 3,
    NON_OBJ: 4
};

torch.jit.mobile.serialization.ObjectType = class ObjectType {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.ObjectType();
        $.type_name = reader.string_(position, 4, null);
        $.type = reader.uint8_(position, 6, 0);
        $.attr_names = reader.strings_(position, 8);
        return $;
    }
};

torch.jit.mobile.serialization.Object = class Object {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.Object();
        $.type_index = reader.uint32_(position, 4, 0);
        $.state = reader.uint32_(position, 6, 0);
        $.attrs = reader.array(position, 8, Uint32Array);
        $.setstate_func = reader.uint32_(position, 10, 0);
        return $;
    }
};

torch.jit.mobile.serialization.ComplexDouble = class ComplexDouble {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.ComplexDouble();
        $.real = reader.float64(position + 0);
        $.imag = reader.float64(position + 4);
        return $;
    }
};

torch.jit.mobile.serialization.EnumValue = class EnumValue {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.EnumValue();
        $.type_name = reader.string_(position, 4, null);
        $.value = reader.uint32_(position, 6, 0);
        return $;
    }
};

torch.jit.mobile.serialization.Instruction = class Instruction {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.Instruction();
        $.op = reader.int8(position + 0);
        $.n = reader.uint16(position + 2);
        $.x = reader.int32(position + 4);
        return $;
    }
};

torch.jit.mobile.serialization.Operator = class Operator {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.Operator();
        $.name = reader.string_(position, 4, null);
        $.overload_name = reader.string_(position, 6, null);
        $.num_args_serialized = reader.int32_(position, 8, -1);
        return $;
    }
};

torch.jit.mobile.serialization.Arg = class Arg {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.Arg();
        $.name = reader.string_(position, 4, null);
        $.type = reader.string_(position, 6, null);
        $.default_value = reader.uint32_(position, 8, 0);
        return $;
    }
};

torch.jit.mobile.serialization.Schema = class Schema {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.Schema();
        $.arguments = reader.tables(position, 4, torch.jit.mobile.serialization.Arg);
        $.returns = reader.tables(position, 6, torch.jit.mobile.serialization.Arg);
        return $;
    }
};

torch.jit.mobile.serialization.DebugInfo = class DebugInfo {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.DebugInfo();
        $.debug_handle = reader.int64s_(position, 4);
        return $;
    }
};

torch.jit.mobile.serialization.Function = class Function {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.Function();
        $.qn = reader.string_(position, 4, null);
        $.instructions = reader.structs(position, 6, torch.jit.mobile.serialization.Instruction);
        $.operators = reader.tables(position, 8, torch.jit.mobile.serialization.Operator);
        $.constants = reader.array(position, 10, Uint32Array);
        $.type_annotations = reader.strings_(position, 12);
        $.register_size = reader.int32_(position, 14, 0);
        $.schema = reader.table(position, 16, torch.jit.mobile.serialization.Schema);
        $.debug_info = reader.table(position, 18, torch.jit.mobile.serialization.DebugInfo);
        $.class_type = reader.uint32_(position, 20, 0);
        return $;
    }
};

torch.jit.mobile.serialization.StorageData = class StorageData {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.StorageData();
        $.data = reader.array(position, 4, Uint8Array);
        return $;
    }
};

torch.jit.mobile.serialization.IValueUnion = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return torch.jit.mobile.serialization.Int.decode(reader, position);
            case 2: return torch.jit.mobile.serialization.Bool.decode(reader, position);
            case 3: return torch.jit.mobile.serialization.Double.decode(reader, position);
            case 4: return torch.jit.mobile.serialization.ComplexDouble.decode(reader, position);
            case 5: return torch.jit.mobile.serialization.TensorMetadata.decode(reader, position);
            case 6: return torch.jit.mobile.serialization.String.decode(reader, position);
            case 7: return torch.jit.mobile.serialization.List.decode(reader, position);
            case 8: return torch.jit.mobile.serialization.Tuple.decode(reader, position);
            case 9: return torch.jit.mobile.serialization.Dict.decode(reader, position);
            case 10: return torch.jit.mobile.serialization.Object.decode(reader, position);
            case 11: return torch.jit.mobile.serialization.IntList.decode(reader, position);
            case 12: return torch.jit.mobile.serialization.DoubleList.decode(reader, position);
            case 13: return torch.jit.mobile.serialization.BoolList.decode(reader, position);
            case 14: return torch.jit.mobile.serialization.Device.decode(reader, position);
            case 15: return torch.jit.mobile.serialization.EnumValue.decode(reader, position);
            case 16: return torch.jit.mobile.serialization.Function.decode(reader, position);
            default: return undefined;
        }
    }
};

torch.jit.mobile.serialization.IValue = class IValue {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.IValue();
        $.val = reader.union(position, 4, torch.jit.mobile.serialization.IValueUnion);
        return $;
    }
};

torch.jit.mobile.serialization.ExtraFile = class ExtraFile {

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.ExtraFile();
        $.name = reader.string_(position, 4, null);
        $.content = reader.string_(position, 6, null);
        return $;
    }
};

torch.jit.mobile.serialization.Module = class Module {

    static identifier(reader) {
        return reader.identifier === 'PTMF';
    }

    static create(reader) {
        return torch.jit.mobile.serialization.Module.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new torch.jit.mobile.serialization.Module();
        $.bytecode_version = reader.uint32_(position, 4, 0);
        $.extra_files = reader.tables(position, 6, torch.jit.mobile.serialization.ExtraFile);
        $.methods = reader.array(position, 8, Uint32Array);
        $.state_obj = reader.uint32_(position, 10, 0);
        $.ivalues = reader.tables(position, 12, torch.jit.mobile.serialization.IValue);
        $.storage_data_size = reader.int32_(position, 14, 0);
        $.storage_data = reader.tables(position, 16, torch.jit.mobile.serialization.StorageData);
        $.object_types = reader.tables(position, 18, torch.jit.mobile.serialization.ObjectType);
        $.jit_sources = reader.tables(position, 20, torch.jit.mobile.serialization.ExtraFile);
        $.jit_constants = reader.array(position, 22, Uint32Array);
        $.operator_version = reader.uint32_(position, 24, 0);
        $.mobile_ivalue_size = reader.uint32_(position, 26, 0);
        return $;
    }
};

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
    QUINT2X4: 17
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

executorch_flatbuffer.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new executorch_flatbuffer.Tensor();
        $.scalar_type = reader.int8_(position, 4, 0);
        $.storage_offset = reader.int32_(position, 6, 0);
        $.sizes = reader.array(position, 8, Int32Array);
        $.dim_order = reader.array(position, 10, Uint8Array);
        $.requires_grad = reader.bool_(position, 12, false);
        $.constant_buffer_idx = reader.uint32_(position, 14, 0);
        $.allocation_info = reader.table(position, 16, executorch_flatbuffer.AllocationDetails);
        $.layout = reader.int8_(position, 18, 0);
        $.shape_dynamism = reader.int8_(position, 20, 0);
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
        return $;
    }
};
