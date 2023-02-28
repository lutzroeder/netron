
var flatbuffers = {};

flatbuffers.get = (name) => {
    flatbuffers._roots = flatbuffers._roots || new Map();
    const roots = flatbuffers._roots;
    if (!roots.has(name)) {
        roots.set(name, {});
    }
    return roots.get(name);
};

flatbuffers.BinaryReader = class {

    static open(data) {
        return data ? new flatbuffers.BinaryReader(data) : null;
    }

    constructor(data) {
        const buffer = data instanceof Uint8Array ? data : data.peek();
        this._buffer = buffer;
        this._position = 0;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }

    get root() {
        return this.int32(this._position) + this._position;
    }

    get identifier() {
        if (this._buffer.length >= 8) {
            const buffer = this._buffer.slice(4, 8);
            if (buffer.every((c) => c >= 32 && c <= 128)) {
                return String.fromCharCode(...buffer);
            }
        }
        return '';
    }

    bool(offset) {
        return !!this.int8(offset);
    }

    bool_(position, offset, defaultValue) {
        offset = this._offset(position, offset);
        return offset ? this.bool(position + offset) : defaultValue;
    }

    int8(offset) {
        return this.uint8(offset) << 24 >> 24;
    }

    int8_(position, offset, defaultValue) {
        offset = this._offset(position, offset);
        return offset ? this.int8(position + offset) : defaultValue;
    }

    uint8(offset) {
        return this._buffer[offset];
    }

    uint8_(position, offset, defaultValue) {
        offset = this._offset(position, offset);
        return offset ? this.uint8(position + offset) : defaultValue;
    }

    int16(offset) {
        return this._dataView.getInt16(offset, true);
    }

    int16_(position, offset, defaultValue) {
        offset = this._offset(position, offset);
        return offset ? this.int16(position + offset) : defaultValue;
    }

    uint16(offset) {
        return this._dataView.getUint16(offset, true);
    }

    uint16_(position, offset, defaultValue) {
        offset = this._offset(position, offset);
        return offset ? this.uint16(position + offset) : defaultValue;
    }

    int32(offset) {
        return this._dataView.getInt32(offset, true);
    }

    int32_(position, offset, defaultValue) {
        offset = this._offset(position, offset);
        return offset ? this.int32(position + offset) : defaultValue;
    }

    uint32(offset) {
        return this._dataView.getUint32(offset, true);
    }

    uint32_(position, offset, defaultValue) {
        offset = this._offset(position, offset);
        return offset ? this.int32(position + offset) : defaultValue;
    }

    int64(offset) {
        return this._dataView.getInt64(offset, true);
    }

    int64_(position, offset, defaultValue) {
        offset = this._offset(position, offset);
        return offset ? this.int64(position + offset) : defaultValue;
    }

    uint64(offset) {
        return this._dataView.getUint64(offset, true);
    }

    uint64_(position, offset, defaultValue) {
        offset = this._offset(position, offset);
        return offset ? this.uint64(position + offset) : defaultValue;
    }

    float32(offset) {
        return this._dataView.getFloat32(offset, true);
    }

    float32_(position, offset, defaultValue) {
        offset = this._offset(position, offset);
        return offset ? this.float32(position + offset) : defaultValue;
    }

    float64(offset) {
        return this._dataView.getFloat64(offset, true);
    }

    float64_(position, offset, defaultValue) {
        offset = this._offset(position, offset);
        return offset ? this.float64(position + offset) : defaultValue;
    }

    string(offset, encoding) {
        offset += this.int32(offset);
        const length = this.int32(offset);
        var result = '';
        var i = 0;
        offset += 4;
        if (encoding === 1) {
            return this._buffer.subarray(offset, offset + length);
        }
        while (i < length) {
            var codePoint;
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
        offset = this._offset(position, offset);
        return offset ? this.string(position + offset) : defaultValue;
    }

    bools_(position, offset) {
        offset = this._offset(position, offset);
        if (offset) {
            const length = this._length(position + offset);
            offset = this._vector(position + offset);
            const array = new Array(length);
            for (let i = 0; i < length; i++) {
                array[i] = this.uint8(offset + i + 4) ? true : false;
            }
            return array;
        }
        return [];
    }

    int64s_(position, offset) {
        offset = this._offset(position, offset);
        if (offset) {
            const length = this._length(position + offset);
            offset = this._vector(position + offset);
            const array = new Array(length);
            for (let i = 0; i < length; i++) {
                array[i] = this.int64(offset + (i << 3));
            }
            return array;
        }
        return [];
    }

    uint64s_(position, offset) {
        offset = this._offset(position, offset);
        if (offset) {
            const length = this._length(position + offset);
            offset = this._vector(position + offset);
            const array = new Array(length);
            for (let i = 0; i < length; i++) {
                array[i] = this.uint64(offset + (i << 3));
            }
            return array;
        }
        return [];
    }

    strings_(position, offset) {
        offset = this._offset(position, offset);
        if (offset) {
            const length = this._length(position + offset);
            offset = this._vector(position + offset);
            const array = new Array(length);
            for (let i = 0; i < length; i++) {
                array[i] = this.string(offset + i * 4);
            }
            return array;
        }
        return [];
    }

    struct(position, offset, decode) {
        offset = this._offset(position, offset);
        return offset ? decode(this, position + offset) : null;
    }

    table(position, offset, decode) {
        offset = this._offset(position, offset);
        return offset ? decode(this, this._indirect(position + offset)) : null;
    }

    union(position, offset, decode) {
        const type_offset = this._offset(position, offset);
        const type = type_offset ? this.uint8(position + type_offset) : 0;
        offset = this._offset(position, offset + 2);
        return offset ? decode(this, this._union(position + offset), type) : null;
    }

    typedArray(position, offset, type) {
        offset = this._offset(position, offset);
        return offset ? new type(this._buffer.buffer, this._buffer.byteOffset + this._vector(position + offset), this._length(position + offset)) : new type(0);
    }

    unionArray(/* position, offset, decode */) {
        return new flatbuffers.Error('Not implemented.');
    }

    structArray(position, offset, decode) {
        offset = this._offset(position, offset);
        const length = offset ? this._length(position + offset) : 0;
        const list = new Array(length);
        for (let i = 0; i < length; i++) {
            list[i] = decode(this, this._vector(position + offset) + i * 8);
        }
        return list;
    }

    tableArray(position, offset, decode) {
        offset = this._offset(position, offset);
        const length = offset ? this._length(position + offset) : 0;
        const list = new Array(length);
        for (let i = 0; i < length; i++) {
            list[i] = decode(this, this._indirect(this._vector(position + offset) + i * 4));
        }
        return list;
    }

    _offset(bb_pos, vtableOffset) {
        var vtable = bb_pos - this.int32(bb_pos);
        return vtableOffset < this.int16(vtable) ? this.int16(vtable + vtableOffset) : 0;
    }

    _indirect(offset) {
        return offset + this.int32(offset);
    }

    _vector(offset) {
        return offset + this.int32(offset) + 4;
    }

    _length(offset) {
        return this.int32(offset + this.int32(offset));
    }

    _union(offset) {
        return offset + this.int32(offset);
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

if (typeof module !== "undefined" && typeof module.exports === "object") {
    module.exports.BinaryReader = flatbuffers.BinaryReader;
    module.exports.TextReader = flatbuffers.TextReader;
    module.exports.get = flatbuffers.get;
}
