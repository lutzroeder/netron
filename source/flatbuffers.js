
const flatbuffers = {};

flatbuffers.BinaryReader = class {

    static open(data, offset) {
        offset = offset || 0;
        if (data && data.length >= (offset + 8)) {
            const reader = data instanceof Uint8Array ?
                new flatbuffers.BinaryReader(data, offset) :
                new flatbuffers.StreamReader(data, offset);
            const root = reader.uint32(offset) + offset;
            if (root < reader.length) {
                const start = root - reader.int32(root);
                if (start > 0 && (start + 4) < reader.length) {
                    const last = reader.int16(start) + start;
                    const max = reader.int16(start + 2);
                    if (last < reader.length) {
                        let valid = true;
                        for (let i = start + 4; i < last; i += 2) {
                            const offset = reader.int16(i);
                            if (offset >= max) {
                                valid = false;
                                break;
                            }
                        }
                        if (valid) {
                            const identifier = reader.identifier;
                            reader.dispose();
                            return new flatbuffers.BinaryReader(data, offset, identifier);
                        }
                    }
                }
            }
            reader.dispose();
        }
        return null;
    }

    constructor(data, offset, identifier) {
        if (data instanceof Uint8Array) {
            this._buffer = data;
            this._dataView = new DataView(data.buffer, data.byteOffset, data.byteLength);
        } else {
            this._data = data;
        }
        this._length = data.length;
        this._offset = offset;
        this._identifier = identifier;
    }

    dispose() {
    }

    get root() {
        if (!this._buffer) {
            this._buffer = this._data.peek();
            this._dataView = new DataView(this._buffer.buffer, this._buffer.byteOffset, this._buffer.byteLength);
            delete this._data;
        }
        return this.int32(this._offset) + this._offset;
    }

    get length() {
        return this._buffer.length;
    }

    get identifier() {
        if (this._identifier === undefined) {
            const buffer = this._buffer.slice(this._offset + 4, this._offset + 8);
            this._identifier = buffer.every((c) => c >= 32 && c <= 128) ? String.fromCharCode(...buffer) : '';
        }
        return this._identifier;
    }

    bool(offset) {
        return !!this.int8(offset);
    }

    bool_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.bool(position + offset) : defaultValue;
    }

    int8(offset) {
        return this.uint8(offset) << 24 >> 24;
    }

    int8_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.int8(position + offset) : defaultValue;
    }

    uint8(offset) {
        return this._buffer[offset];
    }

    uint8_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.uint8(position + offset) : defaultValue;
    }

    int16(offset) {
        return this._dataView.getInt16(offset, true);
    }

    int16_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.int16(position + offset) : defaultValue;
    }

    uint16(offset) {
        return this._dataView.getUint16(offset, true);
    }

    uint16_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.uint16(position + offset) : defaultValue;
    }

    int32(offset) {
        return this._dataView.getInt32(offset, true);
    }

    int32_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.int32(position + offset) : defaultValue;
    }

    uint32(offset) {
        return this._dataView.getUint32(offset, true);
    }

    uint32_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.int32(position + offset) : defaultValue;
    }

    int64(offset) {
        return this._dataView.getBigInt64(offset, true);
    }

    int64_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.int64(position + offset) : defaultValue;
    }

    uint64(offset) {
        return this._dataView.getBigUint64(offset, true);
    }

    uint64_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.uint64(position + offset) : defaultValue;
    }

    float32(offset) {
        return this._dataView.getFloat32(offset, true);
    }

    float32_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.float32(position + offset) : defaultValue;
    }

    float64(offset) {
        return this._dataView.getFloat64(offset, true);
    }

    float64_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.float64(position + offset) : defaultValue;
    }

    string(offset, encoding) {
        offset += this.int32(offset);
        const length = this.int32(offset);
        let result = '';
        let i = 0;
        offset += 4;
        if (encoding === 1) {
            return this._buffer.subarray(offset, offset + length);
        }
        while (i < length) {
            let codePoint;
            // Decode UTF-8
            const a = this.uint8(offset + i++);
            if (a < 0xC0) {
                codePoint = a;
            } else {
                const b = this.uint8(offset + i++);
                if (a < 0xE0) {
                    codePoint = ((a & 0x1F) << 6) | (b & 0x3F);
                } else {
                    const c = this.uint8(offset + i++);
                    if (a < 0xF0) {
                        codePoint = ((a & 0x0F) << 12) | ((b & 0x3F) << 6) | (c & 0x3F);
                    } else {
                        const d = this.uint8(offset + i++);
                        codePoint = ((a & 0x07) << 18) | ((b & 0x3F) << 12) | ((c & 0x3F) << 6) | (d & 0x3F);
                    }
                }
            }
            // Encode UTF-16
            if (codePoint < 0x10000) {
                result += String.fromCharCode(codePoint);
            } else {
                codePoint -= 0x10000;
                result += String.fromCharCode((codePoint >> 10) + 0xD800, (codePoint & ((1 << 10) - 1)) + 0xDC00);
            }
        }

        return result;
    }

    string_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.string(position + offset) : defaultValue;
    }

    bools_(position, offset) {
        offset = this.__offset(position, offset);
        if (offset) {
            const length = this.__vector_len(position + offset);
            offset = this.__vector(position + offset);
            const array = new Array(length);
            for (let i = 0; i < length; i++) {
                array[i] = this.uint8(offset + i + 4) ? true : false;
            }
            return array;
        }
        return [];
    }

    int64s_(position, offset) {
        offset = this.__offset(position, offset);
        if (offset) {
            const length = this.__vector_len(position + offset);
            offset = this.__vector(position + offset);
            const array = new Array(length);
            for (let i = 0; i < length; i++) {
                array[i] = this.int64(offset + (i << 3));
            }
            return array;
        }
        return [];
    }

    uint64s_(position, offset) {
        offset = this.__offset(position, offset);
        if (offset) {
            const length = this.__vector_len(position + offset);
            offset = this.__vector(position + offset);
            const array = new Array(length);
            for (let i = 0; i < length; i++) {
                array[i] = this.uint64(offset + (i << 3));
            }
            return array;
        }
        return [];
    }

    strings_(position, offset) {
        offset = this.__offset(position, offset);
        if (offset) {
            const length = this.__vector_len(position + offset);
            offset = this.__vector(position + offset);
            const array = new Array(length);
            for (let i = 0; i < length; i++) {
                array[i] = this.string(offset + i * 4);
            }
            return array;
        }
        return [];
    }

    struct(position, offset, decode) {
        offset = this.__offset(position, offset);
        return offset ? decode(this, position + offset) : null;
    }

    table(position, offset, decode) {
        offset = this.__offset(position, offset);
        return offset ? decode(this, this.__indirect(position + offset)) : null;
    }

    union(position, offset, decode) {
        const type_offset = this.__offset(position, offset);
        const type = type_offset ? this.uint8(position + type_offset) : 0;
        offset = this.__offset(position, offset + 2);
        return offset ? decode(this, this.__union(position + offset), type) : null;
    }

    typedArray(position, offset, type) {
        offset = this.__offset(position, offset);
        return offset ? new type(this._buffer.buffer, this._buffer.byteOffset + this.__vector(position + offset), this.__vector_len(position + offset)) : new type(0);
    }

    unionArray(/* position, offset, decode */) {
        return new flatbuffers.Error('Not implemented.');
    }

    structArray(position, offset, decode) {
        offset = this.__offset(position, offset);
        const length = offset ? this.__vector_len(position + offset) : 0;
        const list = new Array(length);
        for (let i = 0; i < length; i++) {
            list[i] = decode(this, this.__vector(position + offset) + i * 8);
        }
        return list;
    }

    tableArray(position, offset, decode) {
        offset = this.__offset(position, offset);
        const length = offset ? this.__vector_len(position + offset) : 0;
        const list = new Array(length);
        for (let i = 0; i < length; i++) {
            list[i] = decode(this, this.__indirect(this.__vector(position + offset) + i * 4));
        }
        return list;
    }

    __offset(bb_pos, vtableOffset) {
        const vtable = bb_pos - this.int32(bb_pos);
        return vtableOffset < this.int16(vtable) ? this.int16(vtable + vtableOffset) : 0;
    }

    __indirect(offset) {
        return offset + this.int32(offset);
    }

    __vector(offset) {
        return offset + this.int32(offset) + 4;
    }

    __vector_len(offset) {
        return this.int32(offset + this.int32(offset));
    }

    __union(offset) {
        return offset + this.int32(offset);
    }
};

flatbuffers.StreamReader = class {

    constructor(stream, offset) {
        this._stream = stream;
        this._length = stream.length;
        this._position = stream.position;
        this._offset = offset;
    }

    dispose() {
        this._stream.seek(this._position);
    }

    get length() {
        return this._length;
    }

    get identifier() {
        this._stream.seek(this._offset + 4);
        const buffer = this._stream.peek(4);
        return buffer.every((c) => c >= 32 && c <= 128) ? String.fromCharCode(...buffer) : '';
    }

    int16(offset) {
        this._stream.seek(offset);
        const buffer = this._stream.peek(2);
        return buffer[0] | (buffer[1] << 8);
    }

    int32(offset) {
        this._stream.seek(offset);
        const buffer = this._stream.peek(4);
        return buffer[0] | (buffer[1] << 8) | (buffer[2] << 16) | (buffer[3] << 24);
    }

    uint32(offset) {
        this._stream.seek(offset);
        const buffer = this._stream.peek(4);
        return (buffer[0] | (buffer[1] << 8) | (buffer[2] << 16) | (buffer[3] << 24)) >>> 0;
    }
};

flatbuffers.TextReader = class {

    static open(obj) {
        return new flatbuffers.TextReader(obj);
    }

    constructor(obj) {
        this._root = obj;
    }

    get root() {
        return this._root;
    }

    value(obj, defaultValue) {
        return obj !== undefined ? obj : defaultValue;
    }

    object(obj, decode) {
        return obj !== undefined ? decode(this, obj) : obj;
    }

    array(obj) {
        if (Array.isArray(obj)) {
            const target = new Array(obj.length);
            for (let i = 0; i < obj.length; i++) {
                target[i] = obj[i];
            }
            return target;
        }
        if (!obj) {
            return [];
        }
        throw new flatbuffers.Error('Inalid value array.');
    }

    typedArray(obj, type) {
        if (Array.isArray(obj)) {
            const target = new type(obj.length);
            for (let i = 0; i < obj.length; i++) {
                target[i] = obj[i];
            }
            return target;
        }
        if (!obj) {
            return new type(0);
        }
        throw new flatbuffers.Error('Inalid typed array.');
    }

    objectArray(obj, decode) {
        if (Array.isArray(obj)) {
            const target = new Array(obj.length);
            for (let i = 0; i < obj.length; i++) {
                target[i] = decode(this, obj[i]);
            }
            return target;
        }
        if (!obj) {
            return [];
        }
        throw new flatbuffers.Error('Inalid object array.');
    }
};

flatbuffers.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'FlatBuffers Error';
        this.message = message;
    }
};

export const BinaryReader = flatbuffers.BinaryReader;
export const TextReader = flatbuffers.TextReader;
