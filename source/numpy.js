/* jshint esversion: 6 */

var numpy = numpy || {};

numpy.Array = class {

    constructor(buffer) {
        if (buffer) {
            const reader = new numpy.BinaryReader(buffer);
            const signature = [ 0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59 ];
            if (!reader.read(6).every((v, i) => v == signature[i])) {
                throw new numpy.Error('Invalid signature.');
            }
            const major = reader.byte();
            const minor = reader.byte();
            if (major > 3) {
                throw new numpy.Error("Invalid version '" + [ major, minor ].join('.') + "'.");
            }
            const size = major >= 2 ? reader.uint32() : reader.uint16();
            const encoding = major >= 3 ? 'utf-8' : 'ascii';
            const header_content = new TextDecoder(encoding).decode(reader.read(size));
            const header = numpy.HeaderReader.create(header_content).read();
            if (!header.descr || header.descr.length < 2) {
                throw new numpy.Error("Missing property 'descr'.");
            }
            if (!header.shape) {
                throw new numpy.Error("Missing property 'shape'.");
            }
            this._shape = header.shape;
            this._byteOrder = header.descr[0];
            switch (this._byteOrder) {
                case '|': {
                    this._dataType = header.descr.substring(1);
                    this._data = reader.read(reader.size - reader.position);
                    break;
                }
                case '>':
                case '<': {
                    if (header.descr.length !== 3) {
                        throw new numpy.Error("Unsupported data type '" + header.descr + "'.");
                    }
                    this._dataType = header.descr.substring(1);
                    const size = parseInt(header.descr[2], 10) * this._shape.reduce((a, b) => a * b, 1);
                    this._data = reader.read(size);
                    break;
                }
                default:
                    throw new numpy.Error("Unsupported data type '" + header.descr + "'.");
            }
            if (header.fortran_order) {
                this._data = null;
                // throw new numpy.Error("Fortran order is not supported.'");
            }
        }
    }

    get data() {
        return this._data;
    }

    set data(value) {
        this._data = value;
    }

    get dataType() {
        return this._dataType;
    }

    set dataType(value) {
        this._dataType = value;
    }

    get shape() {
        return this._shape;
    }

    set shape(value) {
        this._shape = value;
    }

    get byteOrder() {
        return this._byteOrder;
    }

    set byteOrder(value) {
        this._byteOrder = value;
    }

    toBuffer() {

        const writer = new numpy.BinaryWriter();

        writer.write([ 0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59 ]); // '\\x93NUMPY'
        writer.byte(1); // major
        writer.byte(0); // minor

        const context = {
            itemSize: 1,
            position: 0,
            dataType: this._dataType,
            byteOrder: this._byteOrder || '<',
            shape: this._shape,
            descr: '',
        };

        if (context.byteOrder !== '<' && context.byteOrder !== '>') {
            throw new numpy.Error("Unknown byte order '" + this._byteOrder + "'.");
        }
        if (context.dataType.length !== 2 || (context.dataType[0] !== 'f' && context.dataType[0] !== 'i' && context.dataType[0] !== 'u')) {
            throw new numpy.Error("Unsupported data type '" + this._dataType + "'.");
        }

        context.itemSize = parseInt(context.dataType[1], 10);

        let shape = '';
        switch (this._shape.length) {
            case 0:
                throw new numpy.Error('Invalid shape.');
            case 1:
                shape = '(' + this._shape[0].toString() + ',)';
                break;
            default:
                shape = '(' + this._shape.map((dimension) => dimension.toString()).join(', ') + ')';
                break;
        }

        const properties = [
            "'descr': '" + context.byteOrder + context.dataType + "'",
            "'fortran_order': False",
            "'shape': " + shape
        ];
        let header = '{ ' + properties.join(', ') + ' }';
        header += ' '.repeat(16 - ((header.length + 2 + 8 + 1) & 0x0f)) + '\n';
        writer.string(header);

        const size = context.itemSize * this._shape.reduce((a, b) => a * b);
        context.data = new Uint8Array(size);
        context.view = new DataView(context.data.buffer, context.data.byteOffset, size);
        numpy.Array._encodeDimension(context, this._data, 0);
        writer.write(context.data);

        return writer.toBuffer();
    }

    static _encodeDimension(context, data, dimension) {
        const size = context.shape[dimension];
        const littleEndian = context.byteOrder === '<';
        if (dimension == context.shape.length - 1) {
            for (let i = 0; i < size; i++) {
                switch (context.dataType) {
                    case 'f2':
                        context.view.setFloat16(context.position, data[i], littleEndian);
                        break;
                    case 'f4':
                        context.view.setFloat32(context.position, data[i], littleEndian);
                        break;
                    case 'f8':
                        context.view.setFloat64(context.position, data[i], littleEndian);
                        break;
                    case 'i1':
                        context.view.setInt8(context.position, data[i], littleEndian);
                        break;
                    case 'i2':
                        context.view.setInt16(context.position, data[i], littleEndian);
                        break;
                    case 'i4':
                        context.view.setInt32(context.position, data[i], littleEndian);
                        break;
                    case 'i8':
                        context.view.setInt64(context.position, data[i], littleEndian);
                        break;
                    case 'u1':
                        context.view.setUint8(context.position, data[i], littleEndian);
                        break;
                    case 'u2':
                        context.view.setUint16(context.position, data[i], littleEndian);
                        break;
                    case 'u4':
                        context.view.setUint32(context.position, data[i], littleEndian);
                        break;
                    case 'u8':
                        context.view.setUint64(context.position, data[i], littleEndian);
                        break;
                }
                context.position += context.itemSize;
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                numpy.Array._encodeDimension(context, data[j], dimension + 1);
            }
        }
    }
};

numpy.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._position = 0;
    }

    get position() {
        return this._position;
    }

    get size() {
        return this._buffer.length;
    }

    byte() {
        return this._buffer[this._position++];
    }

    read(size) {
        const value = this._buffer.slice(this._position, this._position + size);
        this._position += size;
        return value;
    }

    uint16() {
        return this.byte() | (this.byte() << 8);
    }
};

numpy.BinaryWriter = class {

    constructor() {
        this._length = 0;
        this._head = null;
        this._tail = null;
    }

    byte(value) {
        this.write([ value ]);
    }

    uint16(value) {
        this.write([ value & 0xff, (value >> 8) & 0xff ]);
    }

    write(values) {
        const array = new Uint8Array(values.length);
        for (let i = 0; i < values.length; i++) {
            array[i] = values[i];
        }
        this._write(array);
    }

    string(value) {
        this.uint16(value.length);
        const array = new Uint8Array(value.length);
        for (let i = 0; i < value.length; i++) {
            array[i] = value.charCodeAt(i);
        }
        this._write(array);
    }

    _write(array) {
        const node = { buffer: array, next: null };
        if (this._tail) {
            this._tail.next = node;
        }
        else {
            this._head = node;
        }
        this._tail = node;
        this._length += node.buffer.length;
    }

    toBuffer() {
        const array = new Uint8Array(this._length);
        let position = 0;
        let head = this._head;
        while (head != null) {
            array.set(head.buffer, position);
            position += head.buffer.length;
            head = head.next;
        }
        return array;
    }
};

numpy.HeaderReader = class {

    constructor(text) {
        this._text = text;
        this._escape = { '"': '"', '\\': '\\', '/': '/', b: '\b', f: '\f', n: '\n', r: '\r', t: '\t' };
    }

    static create(text) {
        return new numpy.HeaderReader(text);
    }

    read() {
        const decoder = new numpy.HeaderReader.StringDecoder(this._text);
        const stack = [];
        this._decoder = decoder;
        this._position = 0;
        this._char = decoder.decode();
        this._whitespace();
        let obj = undefined;
        let close = undefined;
        let first = true;
        for (;;) {
            this._whitespace();
            let c = this._char;
            if (!first && this._char !== '}' && this._char !== ']' && this._char !== ')') {
                if (this._char !== ',') {
                    this._unexpected();
                }
                this._next();
                this._whitespace();
                c = this._char;
            }
            if (c === close) {
                this._next();
                this._whitespace();
                if (stack.length > 0) {
                    close = stack.pop();
                    obj = stack.pop();
                    first = false;
                    continue;
                }
                if (this._char !== undefined) {
                    this._unexpected();
                }
                return obj;
            }
            first = false;
            let key = undefined;
            if (close === '}') {
                key = this._string();
                switch (key) {
                    case '__proto__':
                    case 'constructor':
                    case 'prototype':
                        throw new numpy.Error("Invalid key '" + key + "'" + this._location());
                }
                this._whitespace();
                if (this._char !== ':') {
                    this._unexpected();
                }
                this._next();
            }
            this._whitespace();
            c = this._char;
            let value = undefined;
            let type = undefined;
            switch (c) {
                case '{': {
                    this._next();
                    value = {};
                    type = '}';
                    first = true;
                    break;
                }
                case '(':
                case '[': {
                    this._next();
                    value = [];
                    type = c === '[' ? ']' : ')';
                    first = true;
                    break;
                }
                default: {
                    value = c === "'" ? this._string() : this._literal();
                    break;
                }
            }
            this._whitespace();
            if (!type && !obj) {
                if (this._char !== undefined) {
                    this._unexpected();
                }
                return value;
            }
            switch (close) {
                case '}':
                    obj[key] = value;
                    break;
                case ')':
                case ']':
                    obj.push(value);
                    break;
            }
            if (type) {
                if (obj) {
                    stack.push(obj);
                    stack.push(close);
                }
                obj = value;
                close = type;
            }
        }
    }

    _next() {
        if (this._char === undefined) {
            this._unexpected();
        }
        this._position = this._decoder.position;
        this._char = this._decoder.decode();
    }

    _whitespace() {
        while (this._char === ' ' || this._char === '\n' || this._char === '\r' || this._char === '\t') {
            this._next();
        }
    }

    _literal() {
        const c = this._char;
        if (c >= '0' && c <= '9') {
            return this._number();
        }
        switch (c) {
            case 'T': this._expect('True'); return true;
            case 'F': this._expect('False'); return false;
            case 'N': this._expect('None'); return null;
            case 'n': this._expect('nan'); return NaN;
            case 'i': this._expect('inf'); return Infinity;
            case '-': return this._number();
        }
        this._unexpected();
    }

    _number() {
        let value = '';
        if (this._char === '-') {
            value = '-';
            this._next();
        }
        if (this._char === 'i') {
            this._expect('inf');
            return -Infinity;
        }
        const c = this._char;
        if (c < '0' || c > '9') {
            this._unexpected();
        }
        value += c;
        this._next();
        if (c === '0') {
            const n = this._char;
            if (n >= '0' && n <= '9') {
                this._unexpected();
            }
        }
        while (this._char >= '0' && this._char <= '9') {
            value += this._char;
            this._next();
        }
        if (this._char === '.') {
            value += '.';
            this._next();
            const n = this._char;
            if (n < '0' || n > '9') {
                this._unexpected();
            }
            while (this._char >= '0' && this._char <= '9') {
                value += this._char;
                this._next();
            }
        }
        if (this._char === 'e' || this._char === 'E') {
            value += this._char;
            this._next();
            const s = this._char;
            if (s === '-' || s === '+') {
                value += this._char;
                this._next();
            }
            const c = this._char;
            if (c < '0' || c > '9') {
                this._unexpected();
            }
            value += this._char;
            this._next();
            while (this._char >= '0' && this._char <= '9') {
                value += this._char;
                this._next();
            }
        }
        return +value;
    }

    _string() {
        let value = '';
        this._next();
        while (this._char != "'") {
            if (this._char === '\\') {
                this._next();
                if (this._char === 'u') {
                    this._next();
                    let uffff = 0;
                    for (let i = 0; i < 4; i ++) {
                        const hex = parseInt(this._char, 16);
                        if (!isFinite(hex)) {
                            this._unexpected();
                        }
                        this._next();
                        uffff = uffff * 16 + hex;
                    }
                    value += String.fromCharCode(uffff);
                }
                else if (this._escape[this._char]) {
                    value += this._escape[this._char];
                    this._next();
                }
                else {
                    this._unexpected();
                }
            }
            else if (this._char < ' ') {
                this._unexpected();
            }
            else {
                value += this._char;
                this._next();
            }
        }
        this._next();
        return value;
    }

    _expect(text) {
        for (let i = 0; i < text.length; i++) {
            if (text[i] !== this._char) {
                this._unexpected();
            }
            this._next();
        }
    }

    _unexpected() {
        let c = this._char;
        if (c === undefined) {
            throw new numpy.Error('Unexpected end of JSON input.');
        }
        if (c < ' ' || c > '\x7F') {
            const name = Object.keys(this._escape).filter((key) => this._escape[key] === c);
            c = (name.length === 1) ? '\\' + name : '\\u' + ('000' + c.charCodeAt(0).toString(16)).slice(-4);
        }
        c = "token '" + c + "'";
        throw new numpy.Error('Unexpected ' + c + this._location());
    }

    _location() {
        let line = 1;
        let column = 1;
        this._decoder.position = 0;
        let c;
        do {
            if (this._decoder.position === this.position) {
                return ' at ' + line.toString() + ':' + column.toString() + '.';
            }
            c = this._decoder.decode();
            if (c === '\n') {
                line++;
                column = 1;
            }
            else {
                column++;
            }
        }
        while (c !== undefined);
        return ' at ' + line.toString() + ':' + column.toString() + '.';
    }
};

numpy.HeaderReader.StringDecoder = class {

    constructor(text) {
        this.text = text;
        this.position = 0;
        this.length = text.length;
    }

    decode() {
        return this.position < this.length ? this.text[this.position++] : undefined;
    }
};

numpy.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'NumPy Error';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Array = numpy.Array;
}