/* jshint esversion: 6 */

var tar = tar || {};

tar.Archive = class {

    constructor(buffer) {
        this._entries = [];
        const reader = buffer instanceof Uint8Array ? new tar.BinaryReader(buffer) : buffer;
        while (reader.position < reader.length) {
            this._entries.push(new tar.Entry(reader));
            if (reader.position + 512 > reader.length ||
                reader.peek(512).every((value) => value === 0x00)) {
                break;
            }
        }
    }

    get entries() {
        return this._entries;
    }
};

tar.Entry = class {

    constructor(reader) {
        const header = reader.peek(512);
        let sum = 0;
        for (let i = 0; i < header.length; i++) {
            sum += (i >= 148 && i < 156) ? 32 : header[i];
        }
        const string = (length) => {
            const buffer = reader.read(length);
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
        };
        this._name = string(100);
        string(8); // file mode
        string(8); // owner
        string(8); // group
        const size = parseInt(string(12).trim(), 8); // size
        string(12); // timestamp
        const checksum = parseInt(string(8).trim(), 8); // checksum
        if (isNaN(checksum) || sum != checksum) {
            throw new tar.Error('Invalid tar archive.');
        }
        string(1); // link indicator
        string(100); // name of linked file
        reader.read(255);
        this._reader = reader.reader(size);
        reader.read(((size % 512) != 0) ? (512 - (size % 512)) : 0);
    }

    get name() {
        return this._name;
    }

    get reader() {
        return this._reader;
    }

    get data() {
        return this.reader.peek();
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

    reader(length) {
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