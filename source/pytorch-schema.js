
import * as flatbuffers from './flatbuffers.js';

const $root = flatbuffers.get('torch');

$root.torch = $root.torch || {};

$root.torch.jit = $root.torch.jit || {};

$root.torch.jit.mobile = $root.torch.jit.mobile || {};

$root.torch.jit.mobile.serialization = $root.torch.jit.mobile.serialization || {};

$root.torch.jit.mobile.serialization.Int = class Int {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.Int();
        $.int_val = reader.int64(position + 0);
        return $;
    }
};

$root.torch.jit.mobile.serialization.Bool = class Bool {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.Bool();
        $.bool_val = reader.bool(position + 0);
        return $;
    }
};

$root.torch.jit.mobile.serialization.Double = class Double {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.Double();
        $.double_val = reader.float64(position + 0);
        return $;
    }
};

$root.torch.jit.mobile.serialization.PerTensorAffineSchema = class PerTensorAffineSchema {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.PerTensorAffineSchema();
        $.q_scale = reader.float64(position + 0);
        $.q_zero_point = reader.int32(position + 4);
        return $;
    }
};

$root.torch.jit.mobile.serialization.QuantizedSchema = class QuantizedSchema {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.QuantizedSchema();
        $.qscheme = reader.int8_(position, 4, 0);
        $.scale = reader.float64_(position, 6, 0);
        $.zero_point = reader.int32_(position, 8, 0);
        $.scales = reader.table(position, 10, $root.torch.jit.mobile.serialization.TensorMetadata.decode);
        $.zero_points = reader.table(position, 12, $root.torch.jit.mobile.serialization.TensorMetadata.decode);
        $.axis = reader.int32_(position, 14, 0);
        return $;
    }
};

$root.torch.jit.mobile.serialization.TensorMetadata = class TensorMetadata {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.TensorMetadata();
        $.storage_location_index = reader.uint32_(position, 4, 0);
        $.scalar_type = reader.int8_(position, 6, 0);
        $.storage_offset = reader.int32_(position, 8, 0);
        $.sizes = reader.typedArray(position, 10, Int32Array);
        $.strides = reader.typedArray(position, 12, Int32Array);
        $.requires_grad = reader.bool_(position, 14, false);
        $.quantized_schema = reader.table(position, 16, $root.torch.jit.mobile.serialization.QuantizedSchema.decode);
        return $;
    }
};

$root.torch.jit.mobile.serialization.String = class String {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.String();
        $.data = reader.string_(position, 4, null);
        return $;
    }
};

$root.torch.jit.mobile.serialization.Device = class Device {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.Device();
        $.str = reader.string_(position, 4, null);
        return $;
    }
};

$root.torch.jit.mobile.serialization.List = class List {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.List();
        $.items = reader.typedArray(position, 4, Uint32Array);
        $.annotation_str = reader.string_(position, 6, null);
        return $;
    }
};

$root.torch.jit.mobile.serialization.IntList = class IntList {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.IntList();
        $.items = reader.int64s_(position, 4);
        return $;
    }
};

$root.torch.jit.mobile.serialization.DoubleList = class DoubleList {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.DoubleList();
        $.items = reader.typedArray(position, 4, Float64Array);
        return $;
    }
};

$root.torch.jit.mobile.serialization.BoolList = class BoolList {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.BoolList();
        $.items = reader.bools_(position, 4);
        return $;
    }
};

$root.torch.jit.mobile.serialization.Tuple = class Tuple {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.Tuple();
        $.items = reader.typedArray(position, 4, Uint32Array);
        return $;
    }
};

$root.torch.jit.mobile.serialization.Dict = class Dict {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.Dict();
        $.keys = reader.typedArray(position, 4, Uint32Array);
        $.values = reader.typedArray(position, 6, Uint32Array);
        $.annotation_str = reader.string_(position, 8, null);
        return $;
    }
};

$root.torch.jit.mobile.serialization.TypeType = {
    UNSET: 0,
    CLASS_WITH_FIELD: 1,
    CUSTOM_CLASS: 2,
    CLASS_WITH_SETSTATE: 3,
    NON_OBJ: 4
};

$root.torch.jit.mobile.serialization.ObjectType = class ObjectType {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.ObjectType();
        $.type_name = reader.string_(position, 4, null);
        $.type = reader.uint8_(position, 6, 0);
        $.attr_names = reader.strings_(position, 8);
        return $;
    }
};

$root.torch.jit.mobile.serialization.Object = class Object {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.Object();
        $.type_index = reader.uint32_(position, 4, 0);
        $.state = reader.uint32_(position, 6, 0);
        $.attrs = reader.typedArray(position, 8, Uint32Array);
        $.setstate_func = reader.uint32_(position, 10, 0);
        return $;
    }
};

$root.torch.jit.mobile.serialization.ComplexDouble = class ComplexDouble {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.ComplexDouble();
        $.real = reader.float64(position + 0);
        $.imag = reader.float64(position + 4);
        return $;
    }
};

$root.torch.jit.mobile.serialization.EnumValue = class EnumValue {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.EnumValue();
        $.type_name = reader.string_(position, 4, null);
        $.value = reader.uint32_(position, 6, 0);
        return $;
    }
};

$root.torch.jit.mobile.serialization.Instruction = class Instruction {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.Instruction();
        $.op = reader.int8(position + 0);
        $.n = reader.uint16(position + 2);
        $.x = reader.int32(position + 4);
        return $;
    }
};

$root.torch.jit.mobile.serialization.Operator = class Operator {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.Operator();
        $.name = reader.string_(position, 4, null);
        $.overload_name = reader.string_(position, 6, null);
        $.num_args_serialized = reader.int32_(position, 8, -1);
        return $;
    }
};

$root.torch.jit.mobile.serialization.Arg = class Arg {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.Arg();
        $.name = reader.string_(position, 4, null);
        $.type = reader.string_(position, 6, null);
        $.default_value = reader.uint32_(position, 8, 0);
        return $;
    }
};

$root.torch.jit.mobile.serialization.Schema = class Schema {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.Schema();
        $.arguments = reader.tableArray(position, 4, $root.torch.jit.mobile.serialization.Arg.decode);
        $.returns = reader.tableArray(position, 6, $root.torch.jit.mobile.serialization.Arg.decode);
        return $;
    }
};

$root.torch.jit.mobile.serialization.DebugInfo = class DebugInfo {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.DebugInfo();
        $.debug_handle = reader.int64s_(position, 4);
        return $;
    }
};

$root.torch.jit.mobile.serialization.Function = class Function {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.Function();
        $.qn = reader.string_(position, 4, null);
        $.instructions = reader.structArray(position, 6, $root.torch.jit.mobile.serialization.Instruction.decode);
        $.operators = reader.tableArray(position, 8, $root.torch.jit.mobile.serialization.Operator.decode);
        $.constants = reader.typedArray(position, 10, Uint32Array);
        $.type_annotations = reader.strings_(position, 12);
        $.register_size = reader.int32_(position, 14, 0);
        $.schema = reader.table(position, 16, $root.torch.jit.mobile.serialization.Schema.decode);
        $.debug_info = reader.table(position, 18, $root.torch.jit.mobile.serialization.DebugInfo.decode);
        $.class_type = reader.uint32_(position, 20, 0);
        return $;
    }
};

$root.torch.jit.mobile.serialization.StorageData = class StorageData {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.StorageData();
        $.data = reader.typedArray(position, 4, Uint8Array);
        return $;
    }
};

$root.torch.jit.mobile.serialization.IValueUnion = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return $root.torch.jit.mobile.serialization.Int.decode(reader, position);
            case 2: return $root.torch.jit.mobile.serialization.Bool.decode(reader, position);
            case 3: return $root.torch.jit.mobile.serialization.Double.decode(reader, position);
            case 4: return $root.torch.jit.mobile.serialization.ComplexDouble.decode(reader, position);
            case 5: return $root.torch.jit.mobile.serialization.TensorMetadata.decode(reader, position);
            case 6: return $root.torch.jit.mobile.serialization.String.decode(reader, position);
            case 7: return $root.torch.jit.mobile.serialization.List.decode(reader, position);
            case 8: return $root.torch.jit.mobile.serialization.Tuple.decode(reader, position);
            case 9: return $root.torch.jit.mobile.serialization.Dict.decode(reader, position);
            case 10: return $root.torch.jit.mobile.serialization.Object.decode(reader, position);
            case 11: return $root.torch.jit.mobile.serialization.IntList.decode(reader, position);
            case 12: return $root.torch.jit.mobile.serialization.DoubleList.decode(reader, position);
            case 13: return $root.torch.jit.mobile.serialization.BoolList.decode(reader, position);
            case 14: return $root.torch.jit.mobile.serialization.Device.decode(reader, position);
            case 15: return $root.torch.jit.mobile.serialization.EnumValue.decode(reader, position);
            case 16: return $root.torch.jit.mobile.serialization.Function.decode(reader, position);
            default: return undefined;
        }
    }
};

$root.torch.jit.mobile.serialization.IValue = class IValue {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.IValue();
        $.val = reader.union(position, 4, $root.torch.jit.mobile.serialization.IValueUnion.decode);
        return $;
    }
};

$root.torch.jit.mobile.serialization.ExtraFile = class ExtraFile {

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.ExtraFile();
        $.name = reader.string_(position, 4, null);
        $.content = reader.string_(position, 6, null);
        return $;
    }
};

$root.torch.jit.mobile.serialization.Module = class Module {

    static identifier(reader) {
        return reader.identifier === 'PTMF';
    }

    static create(reader) {
        return $root.torch.jit.mobile.serialization.Module.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new $root.torch.jit.mobile.serialization.Module();
        $.bytecode_version = reader.uint32_(position, 4, 0);
        $.extra_files = reader.tableArray(position, 6, $root.torch.jit.mobile.serialization.ExtraFile.decode);
        $.methods = reader.typedArray(position, 8, Uint32Array);
        $.state_obj = reader.uint32_(position, 10, 0);
        $.ivalues = reader.tableArray(position, 12, $root.torch.jit.mobile.serialization.IValue.decode);
        $.storage_data_size = reader.int32_(position, 14, 0);
        $.storage_data = reader.tableArray(position, 16, $root.torch.jit.mobile.serialization.StorageData.decode);
        $.object_types = reader.tableArray(position, 18, $root.torch.jit.mobile.serialization.ObjectType.decode);
        $.jit_sources = reader.tableArray(position, 20, $root.torch.jit.mobile.serialization.ExtraFile.decode);
        $.jit_constants = reader.typedArray(position, 22, Uint32Array);
        $.operator_version = reader.uint32_(position, 24, 0);
        $.mobile_ivalue_size = reader.uint32_(position, 26, 0);
        return $;
    }
};

$root.executorch_flatbuffer = $root.executorch_flatbuffer || {};

$root.executorch_flatbuffer.ScalarType = {
    BYTE: 0,
    CHAR: 1,
    SHORT: 2,
    INT: 3,
    LONG: 4,
    FLOAT: 6,
    DOUBLE: 7,
    BOOL: 11,
    QINT8: 12,
    QUINT8: 13,
    QINT32: 14,
    QUINT4X2: 16,
    QUINT2X4: 17
};

$root.executorch_flatbuffer.ContainerMetadata = class ContainerMetadata {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.ContainerMetadata();
        $.encoded_inp_str = reader.string_(position, 4, null);
        $.encoded_out_str = reader.string_(position, 6, null);
        return $;
    }
};

$root.executorch_flatbuffer.Null = class Null {

    static decode(/* reader, position */) {
        const $ = new $root.executorch_flatbuffer.Null();
        return $;
    }
};

$root.executorch_flatbuffer.AllocationDetails = class AllocationDetails {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.AllocationDetails();
        $.memory_id = reader.uint32_(position, 4, 0);
        $.memory_offset = reader.uint32_(position, 6, 0);
        return $;
    }
};

$root.executorch_flatbuffer.TensorShapeDynamism = {
    STATIC: 0,
    DYNAMIC_BOUND: 1,
    DYNAMIC_UNBOUND: 2
};

$root.executorch_flatbuffer.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.Tensor();
        $.scalar_type = reader.int8_(position, 4, 0);
        $.storage_offset = reader.int32_(position, 6, 0);
        $.sizes = reader.typedArray(position, 8, Int32Array);
        $.dim_order = reader.typedArray(position, 10, Uint8Array);
        $.requires_grad = reader.bool_(position, 12, false);
        $.constant_buffer_idx = reader.uint32_(position, 14, 0);
        $.allocation_info = reader.table(position, 16, $root.executorch_flatbuffer.AllocationDetails.decode);
        $.layout = reader.int8_(position, 18, 0);
        $.shape_dynamism = reader.int8_(position, 20, 0);
        return $;
    }
};

$root.executorch_flatbuffer.Int = class Int {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.Int();
        $.int_val = reader.int64_(position, 4, 0);
        return $;
    }
};

$root.executorch_flatbuffer.Bool = class Bool {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.Bool();
        $.bool_val = reader.bool_(position, 4, false);
        return $;
    }
};

$root.executorch_flatbuffer.Double = class Double {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.Double();
        $.double_val = reader.float64_(position, 4, 0);
        return $;
    }
};

$root.executorch_flatbuffer.String = class String {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.String();
        $.string_val = reader.string_(position, 4, null);
        return $;
    }
};

$root.executorch_flatbuffer.IntList = class IntList {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.IntList();
        $.items = reader.int64s_(position, 4);
        return $;
    }
};

$root.executorch_flatbuffer.DoubleList = class DoubleList {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.DoubleList();
        $.items = reader.typedArray(position, 4, Float64Array);
        return $;
    }
};

$root.executorch_flatbuffer.BoolList = class BoolList {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.BoolList();
        $.items = reader.bools_(position, 4);
        return $;
    }
};

$root.executorch_flatbuffer.TensorList = class TensorList {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.TensorList();
        $.items = reader.typedArray(position, 4, Int32Array);
        return $;
    }
};

$root.executorch_flatbuffer.OptionalTensorList = class OptionalTensorList {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.OptionalTensorList();
        $.items = reader.typedArray(position, 4, Int32Array);
        return $;
    }
};

$root.executorch_flatbuffer.KernelTypes = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return $root.executorch_flatbuffer.Null.decode(reader, position);
            case 2: return $root.executorch_flatbuffer.Int.decode(reader, position);
            case 3: return $root.executorch_flatbuffer.Bool.decode(reader, position);
            case 4: return $root.executorch_flatbuffer.Double.decode(reader, position);
            case 5: return $root.executorch_flatbuffer.Tensor.decode(reader, position);
            case 6: return $root.executorch_flatbuffer.String.decode(reader, position);
            case 7: return $root.executorch_flatbuffer.IntList.decode(reader, position);
            case 8: return $root.executorch_flatbuffer.DoubleList.decode(reader, position);
            case 9: return $root.executorch_flatbuffer.BoolList.decode(reader, position);
            case 10: return $root.executorch_flatbuffer.TensorList.decode(reader, position);
            case 11: return $root.executorch_flatbuffer.OptionalTensorList.decode(reader, position);
            default: return undefined;
        }
    }
};

$root.executorch_flatbuffer.EValue = class EValue {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.EValue();
        $.val = reader.union(position, 4, $root.executorch_flatbuffer.KernelTypes.decode);
        return $;
    }
};

$root.executorch_flatbuffer.Operator = class Operator {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.Operator();
        $.name = reader.string_(position, 4, null);
        $.overload = reader.string_(position, 6, null);
        return $;
    }
};

$root.executorch_flatbuffer.KernelCall = class KernelCall {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.KernelCall();
        $.op_index = reader.int32_(position, 4, 0);
        $.args = reader.typedArray(position, 6, Int32Array);
        return $;
    }
};

$root.executorch_flatbuffer.DelegateCall = class DelegateCall {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.DelegateCall();
        $.delegate_index = reader.int32_(position, 4, 0);
        $.args = reader.typedArray(position, 6, Int32Array);
        return $;
    }
};

$root.executorch_flatbuffer.MoveCall = class MoveCall {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.MoveCall();
        $.move_from = reader.int32_(position, 4, 0);
        $.move_to = reader.int32_(position, 6, 0);
        return $;
    }
};

$root.executorch_flatbuffer.JumpFalseCall = class JumpFalseCall {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.JumpFalseCall();
        $.cond_value_index = reader.int32_(position, 4, 0);
        $.destination_instruction = reader.int32_(position, 6, 0);
        return $;
    }
};

$root.executorch_flatbuffer.FreeCall = class FreeCall {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.FreeCall();
        $.value_index = reader.int32_(position, 4, 0);
        return $;
    }
};

$root.executorch_flatbuffer.InstructionArguments = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return $root.executorch_flatbuffer.KernelCall.decode(reader, position);
            case 2: return $root.executorch_flatbuffer.DelegateCall.decode(reader, position);
            case 3: return $root.executorch_flatbuffer.MoveCall.decode(reader, position);
            case 4: return $root.executorch_flatbuffer.JumpFalseCall.decode(reader, position);
            case 5: return $root.executorch_flatbuffer.FreeCall.decode(reader, position);
            default: return undefined;
        }
    }
};

$root.executorch_flatbuffer.Instruction = class Instruction {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.Instruction();
        $.instr_args = reader.union(position, 4, $root.executorch_flatbuffer.InstructionArguments.decode);
        return $;
    }
};

$root.executorch_flatbuffer.Frame = class Frame {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.Frame();
        $.filename = reader.string_(position, 4, null);
        $.lineno = reader.int32_(position, 6, 0);
        $.name = reader.string_(position, 8, null);
        $.context = reader.string_(position, 10, null);
        return $;
    }
};

$root.executorch_flatbuffer.FrameList = class FrameList {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.FrameList();
        $.items = reader.tableArray(position, 4, $root.executorch_flatbuffer.Frame.decode);
        return $;
    }
};

$root.executorch_flatbuffer.DataLocation = {
    INLINE: 0,
    SEGMENT: 1
};

$root.executorch_flatbuffer.BackendDelegateDataReference = class BackendDelegateDataReference {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.BackendDelegateDataReference();
        $.location = reader.int8_(position, 4, 0);
        $.index = reader.uint32_(position, 6, 0);
        return $;
    }
};

$root.executorch_flatbuffer.CompileSpec = class CompileSpec {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.CompileSpec();
        $.key = reader.string_(position, 4, null);
        $.value = reader.typedArray(position, 6, Uint8Array);
        return $;
    }
};

$root.executorch_flatbuffer.BackendDelegate = class BackendDelegate {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.BackendDelegate();
        $.id = reader.string_(position, 4, null);
        $.processed = reader.table(position, 6, $root.executorch_flatbuffer.BackendDelegateDataReference.decode);
        $.compile_specs = reader.tableArray(position, 8, $root.executorch_flatbuffer.CompileSpec.decode);
        return $;
    }
};

$root.executorch_flatbuffer.Chain = class Chain {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.Chain();
        $.inputs = reader.typedArray(position, 4, Int32Array);
        $.outputs = reader.typedArray(position, 6, Int32Array);
        $.instructions = reader.tableArray(position, 8, $root.executorch_flatbuffer.Instruction.decode);
        $.stacktrace = reader.tableArray(position, 10, $root.executorch_flatbuffer.FrameList.decode);
        return $;
    }
};

$root.executorch_flatbuffer.ExecutionPlan = class ExecutionPlan {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.ExecutionPlan();
        $.name = reader.string_(position, 4, null);
        $.container_meta_type = reader.table(position, 6, $root.executorch_flatbuffer.ContainerMetadata.decode);
        $.values = reader.tableArray(position, 8, $root.executorch_flatbuffer.EValue.decode);
        $.inputs = reader.typedArray(position, 10, Int32Array);
        $.outputs = reader.typedArray(position, 12, Int32Array);
        $.chains = reader.tableArray(position, 14, $root.executorch_flatbuffer.Chain.decode);
        $.operators = reader.tableArray(position, 16, $root.executorch_flatbuffer.Operator.decode);
        $.delegates = reader.tableArray(position, 18, $root.executorch_flatbuffer.BackendDelegate.decode);
        $.non_const_buffer_sizes = reader.int64s_(position, 20);
        return $;
    }
};

$root.executorch_flatbuffer.Buffer = class Buffer {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.Buffer();
        $.storage = reader.typedArray(position, 4, Uint8Array);
        return $;
    }
};

$root.executorch_flatbuffer.BackendDelegateInlineData = class BackendDelegateInlineData {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.BackendDelegateInlineData();
        $.data = reader.typedArray(position, 4, Uint8Array);
        return $;
    }
};

$root.executorch_flatbuffer.DataSegment = class DataSegment {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.DataSegment();
        $.offset = reader.uint64_(position, 4, 0);
        $.size = reader.uint64_(position, 6, 0);
        return $;
    }
};

$root.executorch_flatbuffer.SubsegmentOffsets = class SubsegmentOffsets {

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.SubsegmentOffsets();
        $.segment_index = reader.uint32_(position, 4, 0);
        $.offsets = reader.uint64s_(position, 6);
        return $;
    }
};

$root.executorch_flatbuffer.Program = class Program {

    static identifier(reader) {
        return reader.identifier === 'ET12';
    }

    static create(reader) {
        return $root.executorch_flatbuffer.Program.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new $root.executorch_flatbuffer.Program();
        $.version = reader.uint32_(position, 4, 0);
        $.execution_plan = reader.tableArray(position, 6, $root.executorch_flatbuffer.ExecutionPlan.decode);
        $.constant_buffer = reader.tableArray(position, 8, $root.executorch_flatbuffer.Buffer.decode);
        $.backend_delegate_data = reader.tableArray(position, 10, $root.executorch_flatbuffer.BackendDelegateInlineData.decode);
        $.segments = reader.tableArray(position, 12, $root.executorch_flatbuffer.DataSegment.decode);
        $.constant_segment = reader.table(position, 14, $root.executorch_flatbuffer.SubsegmentOffsets.decode);
        return $;
    }
};
