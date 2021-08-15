/* jshint esversion: 6 */

var gzip = gzip || {};
var zip = zip || require('./zip');

gzip.Archive = class {

    static open(data) {
        const stream = data instanceof Uint8Array ? new gzip.BinaryReader(data) : data;
        const signature = [ 0x1f, 0x8b ];
        if (stream.length > 18 && stream.peek(2).every((value, index) => value === signature[index])) {
            return new gzip.Archive(stream);
        }
        return null;
    }

    constructor(stream) {
        const position = stream.position;
        const entry = new gzip.Entry(stream);
        this._entries = new Map([ [ entry.name, entry.stream ] ]);
        stream.seek(position);
    }

    get entries() {
        return this._entries;
    }
};

gzip.Entry = class {

    constructor(stream) {
        const signature = [ 0x1f, 0x8b ];
        if (stream.position + 2 > stream.length ||
            !stream.read(2).every((value, index) => value === signature[index])) {
            throw new gzip.Error('Invalid gzip signature.');
        }
        const string = () => {
            let text = '';
            while (stream.position < stream.length) {
                const value = stream.byte();
                if (value === 0x00) {
                    break;
                }
                text += String.fromCharCode(value);
            }
            return text;
        };
        const reader = new gzip.BinaryReader(stream.read(8));
        const compressionMethod = reader.byte();
        if (compressionMethod != 8) {
            throw new gzip.Error("Invalid compression method '" + compressionMethod.toString() + "'.");
        }
        const flags = reader.byte();
        reader.uint32(); // MTIME
        reader.byte(); // XFL
        reader.byte(); // OS
        if ((flags & 4) != 0) { // FEXTRA
            const xlen = stream.byte() | (stream.byte() << 8);
            stream.skip(xlen);
        }
        this._name = (flags & 8) != 0 ? string() : ''; // FNAME
        if ((flags & 16) != 0) { // FCOMMENT
            string();
        }
        if ((flags & 1) != 0) { // FHCRC
            stream.skip(2);
        }
        this._stream = new gzip.InflaterStream(stream);
    }

    get name() {
        return this._name;
    }

    get stream() {
        return this._stream;
    }
};

gzip.InflaterStream = class {

    constructor(stream) {
        this._stream = stream.stream(stream.length - stream.position - 8);
        const reader = new gzip.BinaryReader(stream.read(8));
        reader.uint32(); // CRC32
        this._length = reader.uint32(); // ISIZE
        this._position = 0;
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    seek(position) {
        if (this._buffer === undefined) {
            this._inflate();
        }
        this._position = position >= 0 ? position : this._length + position;
    }

    skip(offset) {
        if (this._buffer === undefined) {
            this._inflate();
        }
        this._position += offset;
    }

    stream(length) {
        return new gzip.BinaryReader(this.read(length));
    }

    peek(length) {
        const position = this._position;
        length = length !== undefined ? length : this._length - this._position;
        this.skip(length);
        const end = this._position;
        this.seek(position);
        if (position === 0 && length === this._length) {
            return this._buffer;
        }
        return this._buffer.subarray(position, end);
    }

    read(length) {
        const position = this._position;
        length = length !== undefined ? length : this._length - this._position;
        this.skip(length);
        if (position === 0 && length === this._length) {
            return this._buffer;
        }
        return this._buffer.subarray(position, this._position);
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._buffer[position];
    }

    _inflate() {
        if (this._buffer === undefined) {
            const buffer = this._stream.peek();
            this._buffer = new zip.Inflater().inflateRaw(buffer, this._length);
            delete this._stream;
        }
    }
};

gzip.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._length = buffer.length;
        this._position = 0;
        this._view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    create(buffer) {
        return new gzip.BinaryReader(buffer);
    }

    stream(length) {
        return this.create(this.read(length));
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
    }

    skip(offset) {
        this._position += offset;
    }

    peek(length) {
        if (this._position === 0 && length === undefined) {
            return this._buffer;
        }
        const position = this._position;
        this.skip(length !== undefined ? length : this._length - this._position);
        const end = this._position;
        this.seek(position);
        return this._buffer.subarray(position, end);
    }

    read(length) {
        if (this._position === 0 && length === undefined) {
            this._position = this._length;
            return this._buffer;
        }
        const position = this._position;
        this.skip(length !== undefined ? length : this._length - this._position);
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