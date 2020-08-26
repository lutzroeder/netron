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
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.version = reader.int64();
                    break;
                case 4:
                    message.input_shape = reader.array(message.input_shape, () => reader.int64(), tag);
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
$root.dnn.Model.prototype.version = protobuf.Int64.create(0);
$root.dnn.Model.prototype.a014 = 0;

$root.dnn.Parameter = class Parameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.dnn.Parameter();
        const end = reader.next(length);
        while (reader.end(end)) {
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
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dim0 = reader.int64();
                    break;
                case 2:
                    message.dim1 = reader.int64();
                    break;
                case 3:
                    message.dim2 = reader.int64();
                    break;
                case 4:
                    message.dim3 = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.dnn.Shape.prototype.dim0 = protobuf.Int64.create(0);
$root.dnn.Shape.prototype.dim1 = protobuf.Int64.create(0);
$root.dnn.Shape.prototype.dim2 = protobuf.Int64.create(0);
$root.dnn.Shape.prototype.dim3 = protobuf.Int64.create(0);

$root.dnn.Node = class Node {

    constructor() {
        this.input = [];
        this.output = [];
    }

    static decode(reader, length) {
        const message = new $root.dnn.Node();
        const end = reader.next(length);
        while (reader.end(end)) {
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
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = reader.string();
                    break;
                case 3:
                    message.a003 = reader.uint64();
                    break;
                case 7:
                    message.a007 = reader.uint64();
                    break;
                case 8:
                    message.a008 = reader.uint64();
                    break;
                case 9:
                    message.a009 = reader.uint64();
                    break;
                case 10:
                    message.a010 = reader.uint64();
                    break;
                case 11:
                    message.a011 = reader.uint64();
                    break;
                case 14:
                    message.a014 = reader.float();
                    break;
                case 15:
                    message.a015 = reader.float();
                    break;
                case 50:
                    message.weight.push($root.dnn.Tensor.decode(reader, reader.uint32()));
                    break;
                case 72:
                    message.operation = reader.uint64();
                    break;
                case 65:
                    message.axis = reader.uint64();
                    break;
                case 77:
                    message.a077 = reader.uint64();
                    break;
                case 79:
                    message.a079 = reader.float();
                    break;
                case 80:
                    message.a080 = reader.uint64();
                    break;
                case 81:
                    message.a081 = reader.uint64();
                    break;
                case 82:
                    message.a082 = reader.uint64();
                    break;
                case 83:
                    message.a083 = reader.uint64();
                    break;
                case 84:
                    message.a084 = reader.uint64();
                    break;
                case 85:
                    message.a085 = reader.uint64();
                    break;
                case 90:
                    message.a090 = reader.uint64();
                    break;
                case 101:
                    message.a101 = reader.uint64();
                    break;
                case 104:
                    message.a104 = $root.dnn.Buffer.decode(reader, reader.uint32());
                    break;
                case 109:
                    message.a109 = reader.uint64();
                    break;
                case 110:
                    message.a110 = reader.uint64();
                    break;
                case 111:
                    message.a111 = reader.uint64();
                    break;
                case 112:
                    message.a112 = reader.uint64();
                    break;
                case 115:
                    message.a115 = reader.uint64();
                    break;
                case 116:
                    message.a116 = reader.uint64();
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
$root.dnn.Layer.prototype.a003 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a007 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a008 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a009 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a010 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a011 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a014 = 0;
$root.dnn.Layer.prototype.a015 = 0;
$root.dnn.Layer.prototype.operation = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.axis = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a077 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a079 = 0;
$root.dnn.Layer.prototype.a080 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a081 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a082 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a083 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a084 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a085 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a090 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a101 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a104 = null;
$root.dnn.Layer.prototype.a109 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a110 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a111 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a112 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a115 = protobuf.Uint64.create(0);
$root.dnn.Layer.prototype.a116 = protobuf.Uint64.create(0);

$root.dnn.Buffer = class Buffer {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.dnn.Buffer();
        const end = reader.next(length);
        while (reader.end(end)) {
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
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dim0 = reader.int64();
                    break;
                case 2:
                    message.dim1 = reader.int64();
                    break;
                case 3:
                    message.dim2 = reader.int64();
                    break;
                case 4:
                    message.dim3 = reader.int64();
                    break;
                case 5:
                    message.data1 = reader.bytes();
                    break;
                case 6:
                    message.data2 = reader.bytes();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.dnn.Tensor.prototype.dim0 = protobuf.Int64.create(0);
$root.dnn.Tensor.prototype.dim1 = protobuf.Int64.create(0);
$root.dnn.Tensor.prototype.dim2 = protobuf.Int64.create(0);
$root.dnn.Tensor.prototype.dim3 = protobuf.Int64.create(0);
$root.dnn.Tensor.prototype.data1 = new Uint8Array([]);
$root.dnn.Tensor.prototype.data2 = new Uint8Array([]);
