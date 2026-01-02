
export const espdl = {};

espdl.Version = {
    _START_VERSION: 0,
    IR_VERSION_2023_12_22: 1
};

espdl.AttributeType = {
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
    TYPE_FBS: 11,
    TYPE_FBSS: 12
};

espdl.TensorDataType = {
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
    UINT64: 13
};

espdl.DataLocation = {
    DEFAULT: 0,
    EXTERNAL: 1
};

espdl.AttributeF = class AttributeF {

    static decode(reader, position) {
        const $ = new espdl.AttributeF();
        $.f = reader.float32(position + 0);
        return $;
    }
};

espdl.AttributeI = class AttributeI {

    static decode(reader, position) {
        const $ = new espdl.AttributeI();
        $.i = reader.int64(position + 0);
        return $;
    }
};

espdl.Attribute = class Attribute {

    static decode(reader, position) {
        const $ = new espdl.Attribute();
        $.name = reader.string_(position, 4, null);
        $.ref_attr_name = reader.string_(position, 6, null);
        $.doc_string = reader.string_(position, 8, null);
        $.attr_type = reader.int32_(position, 10, 0);
        $.f = reader.struct(position, 12, espdl.AttributeF);
        $.i = reader.struct(position, 14, espdl.AttributeI);
        $.s = reader.array(position, 16, Uint8Array);
        $.t = reader.table(position, 18, espdl.Tensor);
        $.g = reader.table(position, 20, espdl.Graph);
        $.tp = reader.table(position, 22, espdl.TypeInfo);
        $.floats = reader.array(position, 24, Float32Array);
        $.ints = reader.int64s_(position, 26);
        $.strings = reader.strings_(position, 28);
        $.tensors = reader.tables(position, 30, espdl.Tensor);
        $.graphs = reader.tables(position, 32, espdl.Graph);
        $.type_protos = reader.tables(position, 34, espdl.TypeInfo);
        return $;
    }
};

espdl.ValueInfo = class ValueInfo {

    static decode(reader, position) {
        const $ = new espdl.ValueInfo();
        $.name = reader.string_(position, 4, null);
        $.value_info_type = reader.table(position, 6, espdl.TypeInfo);
        $.doc_string = reader.string_(position, 8, null);
        $.exponents = reader.int64s_(position, 10);
        return $;
    }
};

espdl.Node = class Node {

    static decode(reader, position) {
        const $ = new espdl.Node();
        $.input = reader.strings_(position, 4);
        $.output = reader.strings_(position, 6);
        $.name = reader.string_(position, 8, null);
        $.op_type = reader.string_(position, 10, null);
        $.domain = reader.string_(position, 12, null);
        $.attribute = reader.tables(position, 14, espdl.Attribute);
        $.doc_string = reader.string_(position, 16, null);
        return $;
    }
};

espdl.Model = class Model {

    static create(reader) {
        return espdl.Model.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new espdl.Model();
        $.ir_version = reader.int32_(position, 4, 0);
        $.opset_import = reader.tables(position, 6, espdl.OperatorSetId);
        $.producer_name = reader.string_(position, 8, null);
        $.producer_version = reader.string_(position, 10, null);
        $.domain = reader.string_(position, 12, null);
        $.model_version = reader.int64_(position, 14, 0n);
        $.doc_string = reader.string_(position, 16, null);
        $.graph = reader.table(position, 18, espdl.Graph);
        $.metadata_props = reader.tables(position, 20, espdl.StringStringEntry);
        $.functions = reader.tables(position, 22, espdl.Function);
        return $;
    }
};

espdl.StringStringEntry = class StringStringEntry {

    static decode(reader, position) {
        const $ = new espdl.StringStringEntry();
        $.key = reader.string_(position, 4, null);
        $.value = reader.string_(position, 6, null);
        return $;
    }
};

espdl.TensorAnnotation = class TensorAnnotation {

    static decode(reader, position) {
        const $ = new espdl.TensorAnnotation();
        $.tensor_name = reader.string_(position, 4, null);
        $.quant_parameter_tensor_names = reader.tables(position, 6, espdl.StringStringEntry);
        return $;
    }
};

espdl.Graph = class Graph {

    static decode(reader, position) {
        const $ = new espdl.Graph();
        $.node = reader.tables(position, 4, espdl.Node);
        $.name = reader.string_(position, 6, null);
        $.initializer = reader.tables(position, 8, espdl.Tensor);
        $.doc_string = reader.string_(position, 10, null);
        $.input = reader.tables(position, 12, espdl.ValueInfo);
        $.output = reader.tables(position, 14, espdl.ValueInfo);
        $.value_info = reader.tables(position, 16, espdl.ValueInfo);
        $.quantization_annotation = reader.tables(position, 18, espdl.TensorAnnotation);
        $.test_inputs_value = reader.tables(position, 20, espdl.Tensor);
        $.test_outputs_value = reader.tables(position, 22, espdl.Tensor);
        return $;
    }
};

espdl.AlignedBytes = class AlignedBytes {

    static decode() {
        const $ = new espdl.AlignedBytes();
        $.bytes = undefined; // not implemented
        return $;
    }
};

espdl.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new espdl.Tensor();
        $.dims = reader.int64s_(position, 4);
        $.data_type = reader.int32_(position, 6, 0);
        $.float_data = reader.array(position, 8, Float32Array);
        $.int32_data = reader.array(position, 10, Int32Array);
        $.string_data = reader.strings_(position, 12);
        $.int64_data = reader.int64s_(position, 14);
        $.name = reader.string_(position, 16, null);
        $.doc_string = reader.string_(position, 18, null);
        $.raw_data = reader.structs(position, 20, espdl.AlignedBytes);
        $.external_data = reader.tables(position, 22, espdl.StringStringEntry);
        $.data_location = reader.int32_(position, 24, 0);
        $.double_data = reader.array(position, 26, Float64Array);
        $.uint64_data = reader.uint64s_(position, 28);
        $.exponents = reader.int64s_(position, 30);
        return $;
    }
};

espdl.TensorShape = class TensorShape {

    static decode(reader, position) {
        const $ = new espdl.TensorShape();
        $.dim = reader.tables(position, 4, espdl.Dimension);
        return $;
    }
};

espdl.Dimension = class Dimension {

    static decode(reader, position) {
        const $ = new espdl.Dimension();
        $.value = reader.table(position, 4, espdl.DimensionValue);
        $.denotation = reader.string_(position, 6, null);
        return $;
    }
};

espdl.DimensionValueType = {
    UNKNOWN: 0,
    VALUE: 1,
    PARAM: 2
};

espdl.DimensionValue = class DimensionValue {

    static decode(reader, position) {
        const $ = new espdl.DimensionValue();
        $.dim_type = reader.int8_(position, 4, 0);
        $.dim_value = reader.int64_(position, 6, 0n);
        $.dim_param = reader.string_(position, 8, null);
        return $;
    }
};

espdl.TensorTypeAndShape = class TensorTypeAndShape {

    static decode(reader, position) {
        const $ = new espdl.TensorTypeAndShape();
        $.elem_type = reader.int32_(position, 4, 0);
        $.shape = reader.table(position, 6, espdl.TensorShape);
        return $;
    }
};

espdl.SequenceType = class SequenceType {

    static decode(reader, position) {
        const $ = new espdl.SequenceType();
        $.elem_type = reader.table(position, 4, espdl.TypeInfo);
        return $;
    }
};

espdl.MapType = class MapType {

    static decode(reader, position) {
        const $ = new espdl.MapType();
        $.key_type = reader.int32_(position, 4, 0);
        $.value_type = reader.table(position, 6, espdl.TypeInfo);
        return $;
    }
};

espdl.OptionalType = class OptionalType {

    static decode(reader, position) {
        const $ = new espdl.OptionalType();
        $.elem_type = reader.table(position, 4, espdl.TypeInfo);
        return $;
    }
};

espdl.TypeInfoValue = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return espdl.TensorTypeAndShape.decode(reader, position);
            case 2: return espdl.SequenceType.decode(reader, position);
            case 3: return espdl.MapType.decode(reader, position);
            case 4: return espdl.OptionalType.decode(reader, position);
            default: return undefined;
        }
    }
};

espdl.TypeInfo = class TypeInfo {

    static decode(reader, position) {
        const $ = new espdl.TypeInfo();
        $.value = reader.union(position, 4, espdl.TypeInfoValue);
        $.denotation = reader.string_(position, 8, null);
        return $;
    }
};

espdl.OperatorSetId = class OperatorSetId {

    static decode(reader, position) {
        const $ = new espdl.OperatorSetId();
        $.domain = reader.string_(position, 4, null);
        $.version = reader.int64_(position, 6, 0n);
        return $;
    }
};

espdl.Function = class Function {

    static decode(reader, position) {
        const $ = new espdl.Function();
        $.name = reader.string_(position, 4, null);
        $.input = reader.strings_(position, 6);
        $.output = reader.strings_(position, 8);
        $.attribute = reader.strings_(position, 10);
        $.attribute_proto = reader.tables(position, 12, espdl.Attribute);
        $.node = reader.tables(position, 14, espdl.Node);
        $.doc_string = reader.string_(position, 16, null);
        $.opset_import = reader.tables(position, 18, espdl.OperatorSetId);
        $.domain = reader.string_(position, 20, null);
        return $;
    }
};
