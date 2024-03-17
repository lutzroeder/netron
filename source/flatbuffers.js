
const flatbuffers = {};

flatbuffers.BinaryReader = class {

    static open(data, offset) {
        offset = offset || 0;
        if (data && data.length >= (offset + 8)) {
            const position = data instanceof Uint8Array ? -1 : data.position;
            const reader = data instanceof Uint8Array ?
                new flatbuffers.BufferReader(data) :
                new flatbuffers.StreamReader(data);
            reader.root = reader.int32(offset) + offset;
            let value = false;
            if (reader.root > 0 && reader.root < reader.length) {
                const buffer = reader.read(offset + 4, 4);
                reader.identifier = buffer.every((c) => c >= 32 && c <= 128) ? String.fromCharCode(...buffer) : '';
                const start = reader.root - reader.int32(reader.root);
                if (start > 0 && (start + 4) < reader.length) {
                    const last = reader.int16(start) + start;
                    if (last < reader.length) {
                        const max = reader.int16(start + 2);
                        const offsets = [];
                        for (let i = start + 4; i < last; i += 2) {
                            const offset = reader.int16(i);
                            offsets.push(offset);
                        }
                        value = max > Math.max(...offsets);
                    }
                }
            }
            if (position !== -1) {
                data.seek(position);
            }
            if (value) {
                return reader;
            }
        }
        return null;
    }

    bool(offset) {
        return Boolean(this.int8(offset));
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

    uint8_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.uint8(position + offset) : defaultValue;
    }

    int16_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.int16(position + offset) : defaultValue;
    }

    uint16_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.uint16(position + offset) : defaultValue;
    }

    int32_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.int32(position + offset) : defaultValue;
    }

    uint32_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.int32(position + offset) : defaultValue;
    }

    int64_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.int64(position + offset) : defaultValue;
    }

    uint64_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.uint64(position + offset) : defaultValue;
    }

    float32_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.float32(position + offset) : defaultValue;
    }

    float64_(position, offset, defaultValue) {
        offset = this.__offset(position, offset);
        return offset ? this.float64(position + offset) : defaultValue;
    }

    string(offset, encoding) {
        offset += this.int32(offset);
        const length = this.int32(offset);
        offset += 4;
        if (encoding === 1) {
            return this.read(offset, length);
        }
        let text = '';
        for (let i = 0; i < length;) {
            let codePoint;
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
                text += String.fromCharCode(codePoint);
            } else {
                codePoint -= 0x10000;
                text += String.fromCharCode((codePoint >> 10) + 0xD800, (codePoint & ((1 << 10) - 1)) + 0xDC00);
            }
        }
        return text;
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
        if (offset) {
            const length = this.__vector_len(position + offset);
            offset = this.__vector(position + offset);
            const buffer = this.read(offset, length * type.BYTES_PER_ELEMENT);
            return new type(buffer.buffer, buffer.byteOffset, length);
        }
        return new type(0);
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

flatbuffers.BufferReader = class extends flatbuffers.BinaryReader {

    constructor(data) {
        super();
        this.length = data.length;
        this._buffer = data;
        this._view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    }

    read(offset, length) {
        return this._buffer.slice(offset, offset + length);
    }

    uint8(offset) {
        return this._buffer[offset];
    }

    int16(offset) {
        return this._view.getInt16(offset, true);
    }

    uint16(offset) {
        return this._view.getUint16(offset, true);
    }

    int32(offset) {
        return this._view.getInt32(offset, true);
    }

    uint32(offset) {
        return this._view.getUint32(offset, true);
    }

    int64(offset) {
        return this._view.getBigInt64(offset, true);
    }

    uint64(offset) {
        return this._view.getBigUint64(offset, true);
    }

    float32(offset) {
        return this._view.getFloat32(offset, true);
    }

    float64(offset) {
        return this._view.getFloat64(offset, true);
    }
};

flatbuffers.StreamReader = class extends flatbuffers.BinaryReader {

    constructor(stream) {
        super();
        this.length = stream.length;
        this._stream = stream;
        this._fill(0, Math.min(0x100000, stream.length));
    }

    read(offset, length) {
        this._stream.seek(offset);
        return this._stream.read(length);
    }

    uint8(offset) {
        const position = this._fill(offset, 1);
        return this._view.getInt8(position);
    }

    int16(offset) {
        const position = this._fill(offset, 2);
        return this._view.getInt16(position, true);
    }

    uint16(offset) {
        const position = this._fill(offset, 2);
        return this._view.getUint16(position, true);
    }

    int32(offset) {
        const position = this._fill(offset, 4);
        return this._view.getInt32(position, true);
    }

    uint32(offset) {
        const position = this._fill(offset, 4);
        return this._view.getUint32(position, true);
    }

    int64(offset) {
        const position = this._fill(offset, 8);
        return this._view.getBigInt64(position, true);
    }

    uint64(offset) {
        const position = this._fill(offset, 8);
        return this._view.getBigUint64(position, true);
    }

    float32(offset) {
        const position = this._fill(offset, 4);
        return this._view.getFloat32(position, true);
    }

    float64(offset) {
        const position = this._fill(offset, 8);
        return this._view.getFloat64(position, true);
    }

    _fill(offset, length) {
        if (offset + length > this._length) {
            throw new Error(`Expected ${offset + length - this._length} more bytes. The file might be corrupted. Unexpected end of file.`);
        }
        if (!this._buffer || offset < this._offset || offset + length > this._offset + this._buffer.length) {
            this._offset = offset;
            this._stream.seek(this._offset);
            const size = Math.min(0x10000000, this.length - this._offset);
            this._buffer = this._stream.read(size);
            this._view = new DataView(this._buffer.buffer, this._buffer.byteOffset, this._buffer.byteLength);
        }
        return offset - this._offset;
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

    int64(obj, defaultValue) {
        return obj !== undefined ? BigInt(obj) : defaultValue;
    }

    uint64(obj, defaultValue) {
        return obj !== undefined ? BigInt(obj) : defaultValue;
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
