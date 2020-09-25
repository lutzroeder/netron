/* jshint esversion: 6 */
/* global pako */

var gzip = gzip || {};

gzip.Archive = class {

    constructor(buffer) {
        this._entries = [];
        if (buffer.length < 18 || buffer[0] != 0x1f || buffer[1] != 0x8b) {
            throw new gzip.Error('Invalid gzip archive.');
        }
        const reader = new gzip.Reader(buffer, 0, buffer.length);
        this._entries.push(new gzip.Entry(reader));
    }

    get entries() {
        return this._entries;
    }
};

gzip.Entry = class {

    constructor(reader) {
        if (!reader.match([ 0x1f, 0x8b ])) {
            throw new gzip.Error('Invalid gzip signature.');
        }
        const compressionMethod = reader.byte();
        if (compressionMethod != 8) {
            throw new gzip.Error("Invalid compression method '" + compressionMethod.toString() + "'.");
        }
        const flags = reader.byte();
        reader.uint32(); // MTIME
        reader.byte();
        reader.byte(); // OS
        if ((flags & 4) != 0) {
            const xlen = reader.uint16();
            reader.skip(xlen);
        }
        if ((flags & 8) != 0) {
            this._name = reader.string();
        }
        if ((flags & 16) != 0) { // FLG.FCOMMENT
            reader.string();
        }
        if ((flags & 1) != 0) {
            reader.uint16(); // CRC16
        }
        const compressedData = reader.bytes();
        if (typeof process === 'object' && typeof process.versions == 'object' && typeof process.versions.node !== 'undefined') {
            this._data = require('zlib').inflateRawSync(compressedData);
        }
        else if (typeof pako !== 'undefined') {
            this._data = pako.inflateRaw(compressedData);
        }
        else {
            this._data = new require('./zip').Inflater().inflateRaw(compressedData);
        }
        reader.position = -8;
        reader.uint32(); // CRC32
        const size = reader.uint32();
        if (size != this._data.length) {
            throw new gzip.Error('Invalid size.');
        }
    }

    get name() {
        return this._name;
    }

    get data() {
        return this._data;
    }

};

gzip.Reader = class {

    constructor(buffer, start, end) {
        this._buffer = buffer;
        this._position = start;
        this._end = end;
        this._view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }

    match(signature) {
        if (this._position + signature.length <= this._end) {
            for (let i = 0; i < signature.length; i++) {
                if (this._buffer[this._position + i] != signature[i]) {
                    return false;
                }
            }
        }
        this._position += signature.length;
        return true;
    }

    get position() {
        return this._position;
    }

    set position(value) {
        this._position = value >= 0 ? value : this._end + value;
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._end) {
            throw new gzip.Error('Expected ' + (this._position - this._end) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    bytes(size) {
        const position = this._position;
        size = size === undefined ? (this._end - position) : size;
        this.skip(size);
        return this._buffer.subarray(position, this._position);
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._buffer[position];
    }

    uint16() {
        const position = this._position;
        this.skip(2);
        return this._view.getUint16(position, true);
    }

    uint32() {
        const position = this._position;
        this.skip(4);
        return this._view.getUint32(position, true);
    }

    string() {
        let result = '';
        const end = this._buffer.indexOf(0x00, this._position);
        if (end < 0) {
            throw new gzip.Error('End of string not found.');
        }
        while (this._position < end) {
            result += String.fromCharCode(this._buffer[this._position++]);
        }
        this._position++;
        return result;
    }

};

gzip.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Gzip Error';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Archive = gzip.Archive;
}