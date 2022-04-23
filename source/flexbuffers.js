
var flexbuffers = {};

flexbuffers.BinaryReader = class {

    static open(buffer) {
        const length = buffer.length;
        if (length >= 3) {
            const byteWidth = buffer[length - 1];
            if (byteWidth <= 8) {
                const packedType = buffer[length - 2];
                return new flexbuffers.BinaryReader(buffer, length - 2 - byteWidth, byteWidth, 1 << (packedType & 3), packedType >> 2);
            }
        }
        return null;
    }

    constructor(buffer, offset, parentWidth, byteWidth, type) {
        this._buffer = buffer;
        this._length = buffer.length;
        this._view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._utf8Decoder = new TextDecoder('utf-8');
        this._root = new flexbuffers.Reference(this, offset, parentWidth, byteWidth, type);
    }

    read() {
        return this._root.read();
    }

    get length() {
        return this._length;
    }

    int(offset, size) {
        switch (size) {
            case 1: return this._view.getInt8(offset);
            case 2: return this._view.getInt16(offset, true);
            case 4: return this._view.getInt32(offset, true);
            case 8: return this._view.getInt64(offset, true);
            default: throw new flexbuffers.Error("Invalid int size '" + size + "'.");
        }
    }

    uint(offset, size) {
        switch (size) {
            case 1: return this._view.getUint8(offset);
            case 2: return this._view.getUint16(offset, true);
            case 4: return this._view.getUint32(offset, true);
            case 8: return this._view.getUint64(offset, true);
            default: throw new flexbuffers.Error("Invalid uint size '" + size + "'.");
        }
    }

    float(offset, size) {
        switch (size) {
            case 4: return this._view.getFloat32(offset, true);
            case 8: return this._view.getFloat64(offset, true);
            default: throw new flexbuffers.Error("Invalid float size '" + size + "'.");
        }
    }

    string(offset, size) {
        let end = size === undefined ? this._buffer.indexOf(0, offset) : offset + size;
        end = end === -1 ? this._buffer.length : end;
        const bytes = this._buffer.subarray(offset, end);
        return this._utf8Decoder.decode(bytes);
    }

    bytes(offset, size) {
        return this._buffer.slice(offset, offset + size);
    }
};

flexbuffers.Reference = class {

    constructor(reader, offset, parentWidth, byteWidth, type) {
        this._reader = reader;
        this._offset = offset;
        this._parentWidth = parentWidth;
        this._byteWidth = byteWidth;
        this._type = type;
    }

    read() {
        switch (this._type) {
            case 0x00:   // null
                return null;
            case 0x01:   // int
                return this._reader.int(this._offset, this._parentWidth);
            case 0x02:   // uint
                return this._reader.uint(this._offset, this._parentWidth);
            case 0x03:   // float
                return this._reader.float(this._offset, this._parentWidth);
            case 0x04: { // key
                return this._reader.string(this._indirect());
            }
            case 0x05: { // string
                const offset = this._indirect();
                const size = this._reader.uint(offset - this._byteWidth, this._byteWidth);
                return this._reader.string(offset, size);
            }
            case 0x06: // indirect int
                return this._reader.int(this._indirect(), this._byteWidth);
            case 0x07: // indirect uint
                return this._reader.uint(this._indirect(), this._byteWidth);
            case 0x08:   // indirect float
                return this._reader.float(this._indirect(), this._byteWidth);
            case 0x09: { // map
                const offset = this._indirect();
                const keysOffset = offset - (this._byteWidth * 3);
                const keysVectorOffset = keysOffset - this._reader.uint(keysOffset, this._byteWidth);
                const keysByteWidth = this._reader.uint(keysOffset + this._byteWidth, this._byteWidth);
                const keys = this._typedVector(keysVectorOffset, keysByteWidth, 0x04);
                const values = this._vector(offset, this._byteWidth);
                const map = {};
                for (let i = 0; i < keys.length; i++) {
                    map[keys[i]] = values[i];
                }
                return map;
            }
            case 0x0a: { // vector
                return this._vector(this._indirect(), this._byteWidth);
            }
            case 0x0b:   // vector int
            case 0x0c:   // vector uint
            case 0x0d:   // vector float
            case 0x0e:   // vector key
            case 0x0f:   // vector string deprecated
            case 0x24: { // vector bool
                return this._typedVector(this._indirect(), this._byteWidth, this._type - 0x0b + 0x01);
            }
            case 0x10:   // vector int2
            case 0x11:   // vector uint2
            case 0x12:   // vector float2
            case 0x13:   // vector int3
            case 0x14:   // vector uint3
            case 0x15:   // vector float3
            case 0x16:   // vector int4
            case 0x17:   // vector uint4
            case 0x18: { // vector float4
                const offset = this._indirect();
                const size = (((this._type - 0x10) / 3) >> 0) + 2;
                const type = ((this._type - 0x10) % 3) + 0x01;
                return this._typedVector(offset, this._byteWidth, type, size);
            }
            case 0x19: { // blob
                const offset = this._indirect();
                const size = this._reader.uint(offset - this._byteWidth, this._byteWidth);
                return this._reader.bytes(offset, size);
            }
            case 0x1a: { // bool
                return this._reader.uint(this._offset, this._parentWidth) !== 0;
            }
            default: {
                throw new flexbuffers.Error("Unsupported reference type '" + this._type);
            }
        }
    }

    _indirect() {
        return this._offset - this._reader.uint(this._offset, this._parentWidth);
    }

    _vector(offset, byteWidth) {
        const size = this._reader.uint(offset - byteWidth, byteWidth);
        const packedTypeOffset = offset + (size * byteWidth);
        const vector = new Array(size);
        for (let i = 0; i < size; i++) {
            const packedType = this._reader.uint(packedTypeOffset + i, 1);
            const reference = new flexbuffers.Reference(this._reader, offset + (i * byteWidth), byteWidth, 1 << (packedType & 3), packedType >> 2);
            vector[i] = reference.read();
        }
        return vector;
    }

    _typedVector(offset, byteWidth, type, size) {
        size = size === undefined ? this._reader.uint(offset - byteWidth, byteWidth) : size;
        const vector = new Array(size);
        for (let i = 0; i < size; i++) {
            const reference = new flexbuffers.Reference(this._reader, offset + (i * byteWidth), byteWidth, 1, type);
            vector[i] = reference.read();
        }
        return vector;
    }
};

flexbuffers.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'FlexBuffers Error';
        this.message = message;
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.BinaryReader = flexbuffers.BinaryReader;
}