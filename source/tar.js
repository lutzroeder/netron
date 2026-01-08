
const tar = {};

tar.Archive = class {

    static open(data) {
        const stream = data instanceof Uint8Array ? new tar.BinaryReader(data) : data;
        if (stream && stream.length >= 512) {
            const buffer = stream.peek(512);
            const sum = buffer.map((value, index) => (index >= 148 && index < 156) ? 32 : value).reduce((a, b) => a + b, 0);
            const checksum = parseInt(Array.from(buffer.slice(148, 156)).map((c) => String.fromCharCode(c)).join('').split('\0').shift(), 8);
            if (!isNaN(checksum) && sum === checksum) {
                return new tar.Archive(stream);
            }
        }
        return null;
    }

    constructor(stream) {
        this._entries = new Map();
        const position = stream.position;
        while (stream.position < stream.length) {
            const entry = new tar.Entry(stream);
            if (entry.type === '' || entry.type === '0' || entry.type === '1' || entry.type === '2') {
                this._entries.set(entry.name, entry.stream);
            }
            if (stream.position + 512 > stream.length ||
                stream.peek(512).every((value) => value === 0x00)) {
                break;
            }
        }
        stream.seek(position);
    }

    get entries() {
        return this._entries;
    }
};

tar.Entry = class {

    constructor(stream) {
        const buffer = stream.read(512);
        const reader = new tar.BinaryReader(buffer);
        const sum = buffer.map((value, index) => (index >= 148 && index < 156) ? 32 : value).reduce((a, b) => a + b, 0);
        let checksum = '';
        for (let i = 148; i < 156 && buffer[i] !== 0x00; i++) {
            checksum += String.fromCharCode(buffer[i]);
        }
        checksum = parseInt(checksum, 8);
        if (isNaN(checksum) || sum !== checksum) {
            throw new tar.Error('Invalid tar archive.');
        }
        this._name = reader.string(100);
        reader.string(8); // file mode
        reader.string(8); // owner
        reader.string(8); // group
        const size = reader.size();
        reader.string(12); // timestamp
        reader.string(8); // checksum
        this._type = reader.string(1);
        reader.string(100); // name of linked file
        if (reader.string(6) === 'ustar') {
            reader.string(2); // ustar version
            reader.string(32); // owner user name
            reader.string(32); // owner group name
            reader.string(8); // device major number
            reader.string(8); // device number number
            const prefix = reader.string(155);
            this._name = prefix ? `${prefix}/${this._name}` : this._name;
        }
        this._stream = stream.stream(size);
        stream.read(((size % 512) === 0) ? 0 : (512 - (size % 512)));
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get stream() {
        return this._stream;
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
        if (this._position > this._length || this._position < 0) {
            throw new tar.Error('Invalid tar archive. Unexpected end of file.');
        }
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._length || this._position < 0) {
            throw new tar.Error('Invalid tar archive. Unexpected end of file.');
        }
    }

    peek(length) {
        if (this._position === 0 && length === undefined) {
            return this._buffer;
        }
        const position = this._position;
        this.skip(length === undefined ? this._length - this._position : length);
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
        this.skip(length === undefined ? this._length - this._position : length);
        return this._buffer.subarray(position, this._position);
    }

    string(length) {
        const buffer = this.read(length);
        let position = 0;
        let content = '';
        for (let i = 0; i < length; i++) {
            const c = buffer[position++];
            if (c === 0) {
                break;
            }
            content += String.fromCharCode(c);
        }
        return content;
    }

    size() {
        const buffer = this.read(12);
        if (buffer[0] & 0x80) {
            buffer[0] &= 0x7f;
            let value = 0n;
            for (const byte of buffer) {
                value = (value << 8n) | BigInt(byte);
                if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
                    throw new tar.Error('Tar entry size exceeds safe integer.');
                }
            }
            return value.toNumber();
        }
        const octal = String.fromCharCode(...buffer);
        return parseInt(octal.trim() || '0', 8);
    }
};

tar.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'tar Error';
    }
};

export const Archive = tar.Archive;
