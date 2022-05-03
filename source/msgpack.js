
var msgpack = msgpack || {};

// https://github.com/msgpack/msgpack-javascript/blob/master/src/Decoder.ts

msgpack.BinaryReader = class {

    static open(data, callback) {
        return new msgpack.BinaryReader(data, callback);
    }

    constructor(buffer, callback) {
        this._buffer = buffer;
        this._callback = callback;
        this._position = 0;
        this._view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }

    read() {
        const value = this.value();
        return value;
    }

    value() {
        const c = this.byte();
        if (c >= 0xe0) {
            return c - 0x100;
        }
        if (c < 0xC0) {
            if (c < 0x80) {
                return c;
            }
            if (c < 0x90) {
                return this.map(c - 0x80);
            }
            if (c < 0xa0) {
                return this.array(c - 0x90);
            }
            return this.string(c - 0xa0);
        }
        switch (c) {
            case 0xC0: return null;
            case 0xC2: return false;
            case 0xC3: return true;
            case 0xC4: return this.bytes(this.byte());
            case 0xC5: return this.bytes(this.uint16());
            case 0xC6: return this.bytes(this.uint32());
            case 0xC7: return this.extension(this.byte());
            case 0xC8: return this.extension(this.uint16());
            case 0xC9: return this.extension(this.uint32());
            case 0xCA: return this.float32();
            case 0xCB: return this.float64();
            case 0xCC: return this.byte();
            case 0xCD: return this.uint16();
            case 0xCE: return this.uint32();
            case 0xCF: return this.uint64();
            case 0xD0: return this.int8();
            case 0xD1: return this.int16();
            case 0xD2: return this.int32();
            case 0xD3: return this.int64();
            case 0xD4: return this.extension(1);
            case 0xD5: return this.extension(2);
            case 0xD6: return this.extension(4);
            case 0xD7: return this.extension(8);
            case 0xD8: return this.extension(16);
            case 0xD9: return this.string(this.byte());
            case 0xDA: return this.string(this.uint16());
            case 0xDB: return this.string(this.uint32());
            case 0xDC: return this.array(this.uint16());
            case 0xDD: return this.array(this.uint32());
            case 0xDE: return this.map(this.uint16());
            case 0xDF: return this.map(this.uint32());
            default: throw new msgpack.Error("Invalid code '" + c + "'.");
        }
    }

    map(size) {
        const map = {};
        for (let i = 0; i < size; i++) {
            const key = this.value();
            const value = this.value();
            map[key] = value;
        }
        return map;
    }

    array(size) {
        const array = new Array(size);
        for (let i = 0; i < size; i++) {
            array[i] = this.value();
        }
        return array;
    }

    extension(size) {
        const code = this.byte();
        const data = this.bytes(size);
        return this._callback(code, data);
    }

    seek(position) {
        this._position = position;
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new msgpack.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    bytes(size) {
        const data = this._buffer.subarray(this._position, this._position + size);
        this._position += size;
        return data;
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._buffer[position];
    }

    uint16() {
        const position = this._position;
        this.skip(2);
        return this._view.getUint16(position);
    }

    uint32() {
        const position = this._position;
        this.skip(4);
        return this._view.getUint32(position);
    }

    uint64() {
        const position = this._position;
        this.skip(8);
        return this._view.getUint64(position);
    }

    int8() {
        const position = this._position;
        this.skip(1);
        return this._view.getInt8(position);
    }

    int16() {
        const position = this._position;
        this.skip(2);
        return this._view.getInt16(position);
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._view.getInt32(position);
    }

    int64() {
        const position = this._position;
        this.skip(8);
        return this._view.getInt64(position);
    }

    float32() {
        const position = this._position;
        this.skip(4);
        return this._view.getFloat32(position);
    }

    float64() {
        const position = this._position;
        this.skip(8);
        return this._view.getFloat64(position);
    }

    string(size) {
        const buffer = this.bytes(size);
        this._decoder = this._decoder || new TextDecoder('utf8');
        return this._decoder.decode(buffer);
    }
};

msgpack.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'MessagePack Error';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.BinaryReader = msgpack.BinaryReader;
}
