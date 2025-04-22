
import * as fs from 'fs';

const node = {};

node.FileStream = class {

    constructor(file, start, length, mtime) {
        this._file = file;
        this._start = start;
        this._length = length;
        this._position = 0;
        this._mtime = mtime;
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    stream(length) {
        const stream = new node.FileStream(this._file, this._start + this._position, length, this._mtime);
        this.skip(length);
        return stream;
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._length) {
            const offset = this._position - this._length;
            throw new Error(`Expected ${offset} more bytes. The file might be corrupted. Unexpected end of file.`);
        }
    }

    peek(length) {
        length = length === undefined ? this._length - this._position : length;
        if (length < 0x1000000) {
            const position = this._fill(length);
            this._position -= length;
            return this._buffer.subarray(position, position + length);
        }
        const position = this._position;
        this.skip(length);
        this.seek(position);
        const buffer = new Uint8Array(length);
        this._read(buffer, position);
        return buffer;
    }

    read(length) {
        length = length === undefined ? this._length - this._position : length;
        if (length < 0x10000000) {
            const position = this._fill(length);
            return this._buffer.slice(position, position + length);
        }
        const position = this._position;
        this.skip(length);
        const buffer = new Uint8Array(length);
        this._read(buffer, position);
        return buffer;
    }

    _fill(length) {
        if (this._position + length > this._length) {
            const offset = this._position + length - this._length;
            throw new Error(`Expected ${offset} more bytes. The file might be corrupted. Unexpected end of file.`);
        }
        if (!this._buffer || this._position < this._offset || this._position + length > this._offset + this._buffer.length) {
            this._offset = this._position;
            const length = Math.min(0x10000000, this._length - this._offset);
            if (!this._buffer || length !== this._buffer.length) {
                this._buffer = new Uint8Array(length);
            }
            this._read(this._buffer, this._offset);
        }
        const position = this._position;
        this._position += length;
        return position - this._offset;
    }

    _read(buffer, offset) {
        const descriptor = fs.openSync(this._file, 'r');
        const stat = fs.statSync(this._file);
        if (stat.mtimeMs !== this._mtime) {
            throw new Error(`File '${this._file}' last modified time changed.`);
        }
        try {
            fs.readSync(descriptor, buffer, 0, buffer.length, offset + this._start);
        } finally {
            fs.closeSync(descriptor);
        }
    }
};

export const FileStream = node.FileStream;