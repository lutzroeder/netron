/* jshint esversion: 6 */
/* global pako */

var gzip = gzip || {};

gzip.Archive = class {

    // inflate (optional): optimized inflater callback like require('zlib').inflateRawSync or pako.inflateRa
    constructor(buffer) {
        this._entries = [];
        if (buffer.length < 18 || buffer[0] != 0x1f || buffer[1] != 0x8b) {
            throw new gzip.Error('Invalid GZIP archive.');
        }
        var reader = new gzip.Reader(buffer, 0, buffer.length);
        this._entries.push(new gzip.Entry(reader));
    }

    get entries() {
        return this._entries;
    }
};

gzip.Entry = class {

    constructor(reader) {
        if (!reader.match([ 0x1f, 0x8b ])) {
            throw new gzip.Error('Invalid GZIP signature.');
        }
        var compressionMethod = reader.byte();
        if (compressionMethod != 8) {
            throw new gzip.Error("Invalid compression method '" + compressionMethod.toString() + "'.");
        }
        var flags = reader.byte();
        reader.uint32(); // MTIME
        reader.byte();
        reader.byte(); // OS
        if ((flags & 4) != 0) {
            var xlen = reader.uint16();
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
        var compressedData = reader.bytes();
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
        var size = reader.uint32();
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
    }

    match(signature) {
        if (this._position + signature.length <= this._end) {
            for (var i = 0; i < signature.length; i++) {
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

    skip(size) {
        if (this._position + size > this._end) {
            throw new gzip.Error('Data not available.');
        }
        this._position += size;
    }

    bytes(size) {
        if (this._position + size > this._end) {
            throw new gzip.Error('Data not available.');
        }
        size = size === undefined ? this._end : size;
        var data = this._buffer.subarray(this._position, this._position + size);
        this._position += size;
        return data;
    }

    byte() {
        if (this._position + 1 > this._end) {
            throw new gzip.Error('Data not available.');
        }
        var value = this._buffer[this._position];
        this._position++;
        return value;
    }

    uint16() {
        if (this._position + 2 > this._end) {
            throw new gzip.Error('Data not available.');
        }
        var value = this._buffer[this._position] | (this._buffer[this._position + 1] << 8);
        this._position += 2;
        return value;
    }

    uint32() {
        return this.uint16() | (this.uint16() << 16);
    }

    string() {
        var result = '';
        var end = this._buffer.indexOf(0x00, this._position);
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
        this.name = 'gzip Error';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Archive = gzip.Archive;
}