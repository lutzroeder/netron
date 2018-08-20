/*jshint esversion: 6 */

var tar = tar || {};

tar.Archive = class {

    constructor(buffer) {
        this._entries = [];
        var reader = new tar.Reader(buffer, 0, buffer.length);
        while (reader.peek()) {
            this._entries.push(new tar.Entry(reader));
            if (reader.match(512, 0)) {
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
        this._name = reader.readString(100);
        reader.readString(8); // file mode
        reader.readString(8); // owner
        reader.readString(8); // group
        var size = parseInt(reader.readString(12), 8); // size
        reader.readString(12); // timestamp
        reader.readString(8); // checksum
        reader.readString(1); // link indicator
        reader.readString(100); // name of linked file
        reader.read(255);
        this._data = reader.read(size);
        reader.read(((size % 512) != 0) ? (512 - (size % 512)) : 0);
    }

    get name() {
        return this._name;
    }

    get data() {
        return this._data; 
    }
};

tar.Reader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._position = 0;
        this._end = buffer.length;
    }

    peek() {
        return this._position < this._end;
    }

    match(size, value) {
        if (this._position + size <= this._end) {
            if (this._buffer.subarray(this._position, this._position + size).every((c) => c == value)) {
                this._position += size;
                return true;
            }
        }
        return false;
    }

    read(size) {
        if (this._position + size > this._end) {
            throw new tar.Error('Data not available.');
        }
        var data = this._buffer.subarray(this._position, this._position + size);
        this._position += size;
        return data;
    }

    readString(size) {
        var buffer = this.read(size);
        var position = 0;
        var str = '';
        for (var i = 0; i < size; i++) {
            var c = buffer[position++];
            if (c == 0) {
                break;
            }
            str += String.fromCharCode(c);
        }
        return str;
    }
};

tar.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'tar Error';
    }
};
