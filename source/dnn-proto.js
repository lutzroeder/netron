var $root = protobuf.get('dnn');

$root.dnn = {};

$root.dnn.Model = class Model {

    constructor() {
        this.input_shape = [];
        this.input_name = [];
        this.node = [];
        this.input = [];
        this.output = [];
    }

    static decode(reader, length) {
        const message = new $root.dnn.Model();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.version = reader.int32();
                    break;
                case 4:
                    message.input_shape = reader.array(message.input_shape, () => reader.int32(), tag);
                    break;
                case 7:
                    message.input_name.push(reader.string());
                    break;
                case 10:
                    message.node.push($root.dnn.Node.decode(reader, reader.uint32()));
                    break;
                case 12:
                    message.input.push($root.dnn.Parameter.decode(reader, reader.uint32()));
                    break;
                case 13:
                    message.output.push($root.dnn.Parameter.decode(reader, reader.uint32()));
                    break;
                case 14:
                    message.a014 = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.dnn.Model.prototype.name = "";
$root.dnn.Model.prototype.version = 0;
$root.dnn.Model.prototype.a014 = 0;

$root.dnn.Parameter = class Parameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.dnn.Parameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.shape = $root.dnn.Shape.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.dnn.Parameter.prototype.name = "";
$root.dnn.Parameter.prototype.shape = null;

$root.dnn.Shape = class Shape {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.dnn.Shape();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dim0 = reader.int32();
                    break;
                case 2:
                    message.dim1 = reader.int32();
                    break;
                case 3:
                    message.dim2 = reader.int32();
                    break;
                case 4:
                    message.dim3 = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.dnn.Shape.prototype.dim0 = 0;
$root.dnn.Shape.prototype.dim1 = 0;
$root.dnn.Shape.prototype.dim2 = 0;
$root.dnn.Shape.prototype.dim3 = 0;

$root.dnn.Node = class Node {

    constructor() {
        this.input = [];
        this.output = [];
    }

    static decode(reader, length) {
        const message = new $root.dnn.Node();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.layer = $root.dnn.Layer.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.input.push(reader.string());
                    break;
                case 3:
                    message.output.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.dnn.Node.prototype.layer = null;

$root.dnn.Layer = class Layer {

    constructor() {
        this.weight = [];
    }

    static decode(reader, length) {
        const message = new $root.dnn.Layer();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = reader.string();
                    break;
                case 3:
                    message.filters = reader.int32();
                    break;
                case 7:
                    message.a007 = reader.int32();
                    break;
                case 8:
                    message.a008 = reader.int32();
                    break;
                case 9:
                    message.groups = reader.int32();
                    break;
                case 10:
                    message.a010 = reader.int32();
                    break;
                case 11:
                    message.a011 = reader.int32();
                    break;
                case 14:
                    message.slope = reader.float();
                    break;
                case 15:
                    message.intercept = reader.float();
                    break;
                case 50:
                    message.weight.push($root.dnn.Tensor.decode(reader, reader.uint32()));
                    break;
                case 72:
                    message.operation = reader.int32();
                    break;
                case 65:
                    message.axis = reader.int32();
                    break;
                case 77:
                    message.a077 = reader.int32();
                    break;
                case 79:
                    message.scale = reader.float();
                    break;
                case 80:
                    message.pad_1 = reader.int32();
                    break;
                case 81:
                    message.pad_2 = reader.int32();
                    break;
                case 82:
                    message.pad_3 = reader.int32();
                    break;
                case 83:
                    message.pad_4 = reader.int32();
                    break;
                case 84:
                    message.pad_5 = reader.int32();
                    break;
                case 85:
                    message.a085 = reader.int32();
                    break;
                case 90:
                    message.a090 = reader.int32();
                    break;
                case 101:
                    message.is_quantized = reader.bool();
                    break;
                case 104:
                    message.quantization = $root.dnn.Buffer.decode(reader, reader.uint32());
                    break;
                case 109:
                    message.stride_w = reader.int32();
                    break;
                case 110:
                    message.stride_h = reader.int32();
                    break;
                case 111:
                    message.kernel_w = reader.int32();
                    break;
                case 112:
                    message.kernel_h = reader.int32();
                    break;
                case 115:
                    message.a115 = reader.int32();
                    break;
                case 116:
                    message.a116 = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.dnn.Layer.prototype.name = "";
$root.dnn.Layer.prototype.type = "";
$root.dnn.Layer.prototype.filters = 0;
$root.dnn.Layer.prototype.a007 = 0;
$root.dnn.Layer.prototype.a008 = 0;
$root.dnn.Layer.prototype.groups = 0;
$root.dnn.Layer.prototype.a010 = 0;
$root.dnn.Layer.prototype.a011 = 0;
$root.dnn.Layer.prototype.slope = 0;
$root.dnn.Layer.prototype.intercept = 0;
$root.dnn.Layer.prototype.operation = 0;
$root.dnn.Layer.prototype.axis = 0;
$root.dnn.Layer.prototype.a077 = 0;
$root.dnn.Layer.prototype.scale = 0;
$root.dnn.Layer.prototype.pad_1 = 0;
$root.dnn.Layer.prototype.pad_2 = 0;
$root.dnn.Layer.prototype.pad_3 = 0;
$root.dnn.Layer.prototype.pad_4 = 0;
$root.dnn.Layer.prototype.pad_5 = 0;
$root.dnn.Layer.prototype.a085 = 0;
$root.dnn.Layer.prototype.a090 = 0;
$root.dnn.Layer.prototype.is_quantized = false;
$root.dnn.Layer.prototype.quantization = null;
$root.dnn.Layer.prototype.stride_w = 0;
$root.dnn.Layer.prototype.stride_h = 0;
$root.dnn.Layer.prototype.kernel_w = 0;
$root.dnn.Layer.prototype.kernel_h = 0;
$root.dnn.Layer.prototype.a115 = 0;
$root.dnn.Layer.prototype.a116 = 0;

$root.dnn.Buffer = class Buffer {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.dnn.Buffer();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 5:
                    message.data = reader.bytes();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.dnn.Buffer.prototype.data = new Uint8Array([]);

$root.dnn.Tensor = class Tensor {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.dnn.Tensor();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dim0 = reader.int32();
                    break;
                case 2:
                    message.dim1 = reader.int32();
                    break;
                case 3:
                    message.dim2 = reader.int32();
                    break;
                case 4:
                    message.dim3 = reader.int32();
                    break;
                case 5:
                    message.data = reader.bytes();
                    break;
                case 6:
                    message.quantized_data = reader.bytes();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.dnn.Tensor.prototype.dim0 = 0;
$root.dnn.Tensor.prototype.dim1 = 0;
$root.dnn.Tensor.prototype.dim2 = 0;
$root.dnn.Tensor.prototype.dim3 = 0;
$root.dnn.Tensor.prototype.data = new Uint8Array([]);
$root.dnn.Tensor.prototype.quantized_data = new Uint8Array([]);
