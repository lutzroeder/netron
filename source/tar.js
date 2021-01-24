/* jshint esversion: 6 */

var tar = tar || {};

tar.Archive = class {

    static open(buffer) {
        const stream = buffer instanceof Uint8Array ? new tar.BinaryReader(buffer) : buffer;
        if (stream.length > 512) {
            return new tar.Archive(stream);
        }
        throw new tar.Error('Invalid tar archive size.');
    }

    constructor(stream) {
        this._entries = [];
        while (stream.position < stream.length) {
            this._entries.push(new tar.Entry(stream));
            if (stream.position + 512 > stream.length ||
                stream.peek(512).every((value) => value === 0x00)) {
                break;
            }
        }
        stream.seek(0);
    }

    get entries() {
        return this._entries;
    }
};

tar.Entry = class {

    constructor(stream) {
        const buffer = stream.read(512);
        const reader = new tar.BinaryReader(buffer);
        let sum = 0;
        for (let i = 0; i < buffer.length; i++) {
            sum += (i >= 148 && i < 156) ? 32 : buffer[i];
        }
        this._name = reader.string(100);
        reader.string(8); // file mode
        reader.string(8); // owner
        reader.string(8); // group
        const size = parseInt(reader.string(12).trim(), 8); // size
        reader.string(12); // timestamp
        const checksum = parseInt(reader.string(8).trim(), 8); // checksum
        if (isNaN(checksum) || sum != checksum) {
            throw new tar.Error('Invalid tar archive.');
        }
        reader.string(1); // link indicator
        reader.string(100); // name of linked file
        this._stream = stream.stream(size);
        stream.read(((size % 512) != 0) ? (512 - (size % 512)) : 0);
    }

    get name() {
        return this._name;
    }

    get stream() {
        return this._stream;
    }

    get data() {
        return this.stream.peek();
    }
};

tar.BinaryReader = class {

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
        return new tar.BinaryReader(buffer);
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

    string(length) {
        const buffer = this.read(length);
        let position = 0;
        let text = '';
        for (let i = 0; i < length; i++) {
            const c = buffer[position++];
            if (c === 0) {
                break;
            }
            text += String.fromCharCode(c);
        }
        return text;
    }
};

tar.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'tar Error';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Archive = tar.Archive;
}