
/* jshint esversion: 6 */

var protobuf = protobuf || {};
var long = long || { Long: require('long') };

protobuf.get = (name) => {
    protobuf._map = protobuf._map || new Map();
    if (!protobuf._map.has(name)) {
        protobuf._map.set(name, {});
    }
    return protobuf._map.get(name);
};

protobuf.Reader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._length = buffer.length;
        this._position = 0;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._decoder = new TextDecoder('utf-8');
    }

    static create(buffer) {
        return new protobuf.Reader(buffer);
    }

    next(length) {
        if (length === undefined) {
            return this._length;
        }
        return this._position + length;
    }

    end(position) {
        return this._position < position;
    }

    get pos() {
        return this._position;
    }

    string() {
        return this._decoder.decode(this.bytes());
    }

    bool() {
        return this.uint32() !== 0;
    }

    bytes() {
        const length = this.uint32();
        const start = this._position;
        const end = this._position + length;
        if (end > this._length)
            throw this._indexOutOfRangeError(length);
        this._position += length;
        return this._buffer.slice(start, end);
    }

    uint32() {
        let value = 4294967295;
        value = (this._buffer[this._position] & 127) >>> 0;
        if (this._buffer[this._position++] < 128) {
            return value;
        }
        value = (value | (this._buffer[this._position] & 127) <<  7) >>> 0;
        if (this._buffer[this._position++] < 128) return value;
        value = (value | (this._buffer[this._position] & 127) << 14) >>> 0; if (this._buffer[this._position++] < 128) return value;
        value = (value | (this._buffer[this._position] & 127) << 21) >>> 0; if (this._buffer[this._position++] < 128) return value;
        value = (value | (this._buffer[this._position] &  15) << 28) >>> 0; if (this._buffer[this._position++] < 128) return value;
        if ((this._position += 5) > this._length) {
            this._position = this._length;
            throw this._indexOutOfRangeError(10);
        }
        return value;
    }

    int32() {
        return this.uint32() | 0;
    }

    sint32() {
        const value = this.uint32();
        return value >>> 1 ^ -(value & 1) | 0;
    }

    int64() {
        return this._readLongVarint().toLong(false);
    }

    uint64() {
        return this._readLongVarint().toLong(true);
    }

    sint64() {
        return this._readLongVarint().zzDecode().toLong(false);
    }

    fixed64() {
        return this._readFixed64().toLong(true);
    }

    sfixed64() {
        return this._readFixed64().toLong(false);
    }

    fixed32() {
        if (this._position + 4 > this._length) {
            throw this._indexOutOfRangeError(4);
        }
        this._position += 4;
        return this._readFixed32();
    }

    sfixed32() {
        return this.fixed32() | 0;
    }

    float() {
        if (this._position + 4 > this._length) {
            throw this._indexOutOfRangeError(4);
        }
        const position = this._position;
        this._position += 4;
        return this._dataView.getFloat32(position, true);
    }

    double() {
        if (this._position + 8 > this._length) {
            throw this._indexOutOfRangeError(4);
        }
        const position = this._position;
        this._position += 8;
        return this._dataView.getFloat64(position, true);
    }

    array(obj, item, tag) {
        if ((tag & 7) === 2) {
            const end = this.uint32() + this._position;
            while (this._position < end) {
                obj.push(item());
            }
        }
        else {
            obj.push(item());
        }
        return obj;
    }

    floats(obj, tag) {
        if ((tag & 7) === 2) {
            if (obj && obj.length > 0) {
                throw new protobuf.Error('Invalid packed float array.');
            }
            const size = this.uint32();
            const end = this._position + size;
            const length = size >>> 2;
            obj = size > 1048576 ? new Float32Array(length) : new Array(length);
            let position = this._position;
            for (let i = 0; i < length; i++) {
                obj[i] = this._dataView.getFloat32(position, true);
                position += 4;
            }
            this._position = end;
        }
        else {
            if (obj !== undefined && obj.length < 1000000) {
                obj.push(this.float());
            }
            else {
                obj = undefined;
                this.float();
            }
        }
        return obj;
    }

    doubles(obj, tag) {
        if ((tag & 7) === 2) {
            if (obj && obj.length > 0) {
                throw new protobuf.Error('Invalid packed float array.');
            }
            const size = this.uint32();
            const end = this._position + size;
            const length = size >>> 3;
            obj = size > 1048576 ? new Float64Array(length) : new Array(length);
            let position = this._position;
            for (let i = 0; i < length; i++) {
                obj[i] = this._dataView.getFloat64(position, true);
                position += 8;
            }
            this._position = end;
        }
        else {
            if (obj !== undefined && obj.length < 1000000) {
                obj.push(this.double());
            }
            else {
                obj = undefined;
                this.double();
            }
        }
        return obj;
    }

    skip(length) {
        if (typeof length === 'number') {
            if (this._position + length > this._length) {
                throw this._indexOutOfRangeError(length);
            }
            this._position += length;
        }
        else {
            do {
                if (this._position >= this._length) {
                    throw this._indexOutOfRangeError();
                }
            }
            while (this._buffer[this._position++] & 128);
        }
        return this;
    }

    skipType(wireType) {
        switch (wireType) {
            case 0:
                this.skip();
                break;
            case 1:
                this.skip(8);
                break;
            case 2:
                this.skip(this.uint32());
                break;
            case 3:
                while ((wireType = this.uint32() & 7) !== 4) {
                    this.skipType(wireType);
                }
                break;
            case 5:
                this.skip(4);
                break;
            default:
                throw new protobuf.Error('invalid wire type ' + wireType + ' at offset ' + this._position);
        }
    }

    pair(obj, key, value) {
        this.skip();
        this._position++;
        const k = typeof key === 'object' ? protobuf.LongBits.hash(key()) : key();
        this._position++;
        const v = value();
        obj[k] = v;
    }

    _readFixed32() {
        return (this._buffer[this._position - 4] | this._buffer[this._position - 3] << 8 | this._buffer[this._position - 2] << 16 | this._buffer[this._position - 1] << 24) >>> 0;
    }

    _readFixed64() {
        if (this._position + 8 > this._length) {
            throw this._indexOutOfRangeError(8);
        }
        this._position += 4;
        const lo = this._readFixed32();
        this._position += 4;
        const hi = this._readFixed32();
        return new protobuf.LongBits(lo, hi);
    }

    _readLongVarint() {
        const bits = new protobuf.LongBits(0, 0);
        let i = 0;
        if (this._length - this._position > 4) { // fast route (lo)
            for (; i < 4; ++i) {
                // 1st..4th
                bits.lo = (bits.lo | (this._buffer[this._position] & 127) << i * 7) >>> 0;
                if (this._buffer[this._position++] < 128)
                    return bits;
            }
            // 5th
            bits.lo = (bits.lo | (this._buffer[this._position] & 127) << 28) >>> 0;
            bits.hi = (bits.hi | (this._buffer[this._position] & 127) >>  4) >>> 0;
            if (this._buffer[this._position++] < 128)
                return bits;
            i = 0;
        }
        else {
            for (; i < 3; ++i) {
                if (this._position >= this._length)
                    throw this._indexOutOfRangeError();
                bits.lo = (bits.lo | (this._buffer[this._position] & 127) << i * 7) >>> 0;
                if (this._buffer[this._position++] < 128)
                    return bits;
            }
            bits.lo = (bits.lo | (this._buffer[this._position++] & 127) << i * 7) >>> 0;
            return bits;
        }
        if (this._length - this._position > 4) {
            for (; i < 5; ++i) {
                bits.hi = (bits.hi | (this._buffer[this._position] & 127) << i * 7 + 3) >>> 0;
                if (this._buffer[this._position++] < 128)
                    return bits;
            }
        }
        else {
            for (; i < 5; ++i) {
                if (this._position >= this._length) {
                    throw this._indexOutOfRangeError();
                }
                bits.hi = (bits.hi | (this._buffer[this._position] & 127) << i * 7 + 3) >>> 0;
                if (this._buffer[this._position++] < 128)
                    return bits;
            }
        }
        throw new protobuf.Error('Invalid varint encoding.');
    }

    _indexOutOfRangeError(length) {
        return RangeError('index out of range: ' + this.pos + ' + ' + (length || 1) + ' > ' + this.len);
    }
};

protobuf.TextReader = class {

    constructor(text) {
        this._text = text;
        this._position = 0;
        this._lineEnd = -1;
        this._lineStart = 0;
        this._line = -1;
        this._depth = 0;
        this._arrayDepth = 0;
        this._token = '';
    }

    static create(text) {
        return new protobuf.TextReader(text);
    }

    start() {
        if (this._depth > 0) {
            this.expect('{');
        }
        this._depth++;
    }

    end() {
        const token = this.peek();
        if (this._depth > 0 && token === '}') {
            this.expect('}');
            this.match(';');
            this._depth--;
            return true;
        }
        return token === '';
    }

    tag() {
        const name = this.read();
        const separator = this.peek();
        if (separator !== '[' && separator !== '{') {
            this.expect(':');
        }
        return name;
    }

    assert(tag) {
        const token = this.tag();
        if (token !== tag) {
            throw new protobuf.Error("Unexpected '" + token + "' instead of '" + tag + "'" + this.location());
        }
    }

    integer() {
        const token = this.read();
        const value = Number.parseInt(token, 10);
        if (Number.isNaN(token - value)) {
            throw new protobuf.Error("Couldn't parse integer '" + token + "'" + this.location());
        }
        this.semicolon();
        return value;
    }

    float() {
        let token = this.read();
        if (token.startsWith('nan')) {
            return NaN;
        }
        if (token.startsWith('inf')) {
            return Infinity;
        }
        if (token.startsWith('-inf')) {
            return -Infinity;
        }
        if (token.endsWith('f')) {
            token = token.substring(0, token.length - 1);
        }
        const value = Number.parseFloat(token);
        if (Number.isNaN(token - value)) {
            throw new protobuf.Error("Couldn't parse float '" + token + "'" + this.location());
        }
        this.semicolon();
        return value;
    }

    string() {
        const token = this.read();
        if (token.length < 2) {
            throw new protobuf.Error('String is too short' + this.location());
        }
        const quote = token[0];
        if (quote !== "'" && quote !== '"') {
            throw new protobuf.Error('String is not in quotes' + this.location());
        }
        if (quote !== token[token.length - 1]) {
            throw new protobuf.Error('String quotes do not match' + this.location());
        }
        const value = token.substring(1, token.length - 1);
        this.semicolon();
        return value;
    }

    boolean() {
        const token = this.read();
        switch (token) {
            case 'true':
            case 'True':
            case '1':
                this.semicolon();
                return true;
            case 'false':
            case 'False':
            case '0':
                this.semicolon();
                return false;
        }
        throw new protobuf.Error("Couldn't parse boolean '" + token + "'" + this.location());
    }

    bytes() {
        const token = this.string();
        let i = 0;
        let o = 0;
        const length = token.length;
        const a = new Uint8Array(length);
        while (i < length) {
            let c = token.charCodeAt(i++);
            if (c !== 0x5C) {
                a[o++] = c;
            }
            else {
                if (i >= length) {
                    throw new protobuf.Error('Unexpected end of bytes string' + this.location());
                }
                c = token.charCodeAt(i++);
                switch (c) {
                    case 0x27: a[o++] = 0x27; break; // '
                    case 0x5C: a[o++] = 0x5C; break; // \\
                    case 0x22: a[o++] = 0x22; break; // "
                    case 0x72: a[o++] = 0x0D; break; // \r
                    case 0x6E: a[o++] = 0x0A; break; // \n
                    case 0x74: a[o++] = 0x09; break; // \t
                    case 0x62: a[o++] = 0x08; break; // \b
                    case 0x58: // x
                    case 0x78: // X
                        for (let xi = 0; xi < 2; xi++) {
                            if (i >= length) {
                                throw new protobuf.Error('Unexpected end of bytes string' + this.location());
                            }
                            let xd = token.charCodeAt(i++);
                            xd = xd >= 65 && xd <= 70 ? xd - 55 : xd >= 97 && xd <= 102 ? xd - 87 : xd >= 48 && xd <= 57 ? xd - 48 : -1;
                            if (xd === -1) {
                                throw new protobuf.Error("Unexpected hex digit '" + xd + "' in bytes string" + this.location());
                            }
                            a[o] = a[o] << 4 | xd;
                        }
                        o++;
                        break;
                    default:
                        if (c < 48 || c > 57) { // 0-9
                            throw new protobuf.Error("Unexpected character '" + c + "' in bytes string" + this.location());
                        }
                        i--;
                        for (let oi = 0; oi < 3; oi++) {
                            if (i >= length) {
                                throw new protobuf.Error('Unexpected end of bytes string' + this.location());
                            }
                            const od = token.charCodeAt(i++);
                            if (od < 48 || od > 57) {
                                throw new protobuf.Error("Unexpected octal digit '" + od + "' in bytes string" + this.location());
                            }
                            a[o] = a[o] << 3 | od - 48;
                        }
                        o++;
                        break;
                }
            }
        }
        return a.slice(0, o);
    }

    enum(type) {
        const token = this.read();
        if (!Object.prototype.hasOwnProperty.call(type, token)) {
            const value = Number.parseInt(token, 10);
            if (!Number.isNaN(token - value)) {
                this.semicolon();
                return value;
            }
            throw new protobuf.Error("Couldn't parse enum '" + token + "'" + this.location());
        }
        this.semicolon();
        return type[token];
    }

    any(message) {
        if (this.match('[')) {
            this.read();
            const begin = this._position;
            const end = this._text.indexOf(']', begin);
            if (end === -1 || end >= this.next) {
                throw new protobuf.Error('End of Any type_url not found' + this.location());
            }
            message.type_url = this._text.substring(begin, end);
            this._position = end + 1;
            message.value = this.skip().substring(1);
            this.expect('}');
            this.match(';');
            return true;
        }
        return false;
    }

    pair(obj, key, value) {
        this.start();
        let k;
        let v;
        while (!this.end()) {
            switch (this.tag()) {
                case 'key':
                    k = key();
                    break;
                case 'value':
                    v = value();
                    break;
            }
        }
        obj[k] = v;
    }

    array(obj, item) {
        if (this.first()) {
            while (!this.last()) {
                obj.push(item());
                this.next();
            }
        }
        else {
            obj.push(item());
        }
    }

    first() {
        if (this.match('[')) {
            this._arrayDepth++;
            return true;
        }
        return false;
    }

    last() {
        if (this.match(']')) {
            this._arrayDepth--;
            return true;
        }
        return false;
    }

    next() {
        const token = this.peek();
        if (token === ',') {
            this.read();
            return;
        }
        if (token === ']') {
            return;
        }
        this.handle(token);
    }

    skip() {
        let token = this.peek();
        if (token === '{') {
            const message = this._position;
            const depth = this._depth;
            this.start();
            while (!this.end() || depth < this._depth) {
                token = this.peek();
                if (token === '{') {
                    this.start();
                }
                else if (token !== '}') {
                    this.read();
                    this.match(';');
                }
            }
            return this._text.substring(message, this._position);
        }
        else if (token === '[') {
            const list = this._position;
            this.read();
            while (!this.last()) {
                token = this.read();
                if (token === '') {
                    this.handle(token);
                }
            }
            return this._text.substring(list, this._position);
        }
        const position = this._position;
        this.read();
        this.semicolon();
        return this._text.substring(position, this._position);
    }

    handle(token) {
        throw new protobuf.Error("Unexpected token '" + token + "'" + this.location());
    }

    field(token /*, module */) {
        throw new protobuf.Error("Unknown field '" + token + "'" + this.location());
    }

    whitespace() {
        for (;;) {
            while (this._position >= this._lineEnd) {
                this._lineStart = this._lineEnd + 1;
                this._position = this._lineStart;
                if (this._position >= this._text.length) {
                    return false;
                }
                this._lineEnd = this._text.indexOf('\n', this._position);
                if (this._lineEnd === -1) {
                    this._lineEnd = this._text.length;
                }
                this._line++;
            }
            const c = this._text[this._position];
            switch (c) {
                case ' ':
                case '\r':
                case '\t':
                    this._position++;
                    break;
                case '#':
                    this._position = this._lineEnd;
                    break;
                default:
                    return true;
            }
        }
    }

    tokenize() {
        if (!this.whitespace()) {
            this._token = '';
            return this._token;
        }
        let c = this._text[this._position];
        if (c === '[' && this._position + 2 < this._lineEnd) {
            let i = this._position + 1;
            let x = this._text[i];
            if (x >= 'a' && x <= 'z' || x >= 'A' && x <= 'Z') {
                i++;
                while (i < this._lineEnd) {
                    x = this._text[i];
                    i++;
                    if (x >= 'a' && x <= 'z' || x >= 'A' && x <= 'Z' || x >= '0' && x <= '9' || x === '.' || x === '/') {
                        continue;
                    }
                    if (x === ']') {
                        this._token = this._text.substring(this._position, i);
                        return this._token;
                    }
                }
            }
        }
        if (c === '{' || c === '}' || c === ':' || c === '[' || c === ',' || c === ']' || c === ';') {
            this._token = c;
            return this._token;
        }
        let position = this._position + 1;
        if (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c === '_' || c === '$') {
            while (position < this._lineEnd) {
                c = this._text[position];
                if (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c === '_' || c === '+' || c === '-') {
                    position++;
                    continue;
                }
                break;
            }
            this._token = this._text.substring(this._position, position);
            return this._token;
        }
        if (c >= '0' && c <= '9' || c === '-' || c === '+' || c === '.') {
            while (position < this._lineEnd) {
                c = this._text[position];
                if (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c === '_' || c === '+' || c === '-' || c === '.') {
                    position++;
                    continue;
                }
                break;
            }
            this._token = this._text.substring(this._position, position);
            return this._token;
        }
        if (c === '"' || c === "'") {
            const quote = c;
            while (position < this._lineEnd) {
                c = this._text[position];
                if (c === '\\' && position < this._lineEnd) {
                    position += 2;
                    continue;
                }
                position++;
                if (c === quote) {
                    break;
                }
            }
            this._token = this._text.substring(this._position, position);
            return this._token;
        }
        throw new protobuf.Error("Unexpected token '" + c + "'" + this.location());
    }

    peek() {
        if (!this._cache) {
            this._token = this.tokenize();
            this._cache = true;
        }
        return this._token;
    }

    read() {
        if (!this._cache) {
            this._token = this.tokenize();
        }
        this._position += this._token.length;
        this._cache = false;
        return this._token;
    }

    expect(value) {
        const token = this.read();
        if (token !== value) {
            throw new protobuf.Error("Unexpected '" + token + "' instead of '" + value + "'" + this.location());
        }
    }

    match(value) {
        if (this.peek() === value) {
            this.read();
            return true;
        }
        return false;
    }

    semicolon() {
        if (this._arrayDepth === 0) {
            this.match(';');
        }
    }

    location() {
        return ' at ' + (this._line + 1).toString() + ':' + (this._position - this._lineStart + 1).toString();
    }
};

protobuf.Long = long.Long;

protobuf.LongBits = class {

    constructor(lo, hi) {
        this.lo = lo >>> 0;
        this.hi = hi >>> 0;
    }

    toLong(unsigned) {
        return protobuf.Long
            ? new protobuf.Long(this.lo | 0, this.hi | 0, unsigned)
            : { low: this.lo | 0, high: this.hi | 0, unsigned: unsigned };
    }

    toNumber(unsigned) {
        if (!unsigned && this.hi >>> 31) {
            const lo = ~this.lo + 1 >>> 0;
            let hi = ~this.hi     >>> 0;
            if (!lo) {
                hi = hi + 1 >>> 0;
            }
            return -(lo + hi * 4294967296);
        }
        return this.lo + this.hi * 4294967296;
    }

    toHash() {
        return String.fromCharCode(
            this.lo        & 255,
            this.lo >>> 8  & 255,
            this.lo >>> 16 & 255,
            this.lo >>> 24      ,
            this.hi        & 255,
            this.hi >>> 8  & 255,
            this.hi >>> 16 & 255,
            this.hi >>> 24
        );
    }

    zzDecode() {
        const mask = -(this.lo & 1);
        this.lo  = ((this.lo >>> 1 | this.hi << 31) ^ mask) >>> 0;
        this.hi  = ( this.hi >>> 1                  ^ mask) >>> 0;
        return this;
    }

    from(value) {
        if (typeof value === 'number') {
            return protobuf.LongBits.fromNumber(value);
        }
        if (typeof value === 'string' || value instanceof String) {
            if (!protobuf.Long) {
                return protobuf.LongBits.fromNumber(parseInt(value, 10));
            }
            value = protobuf.Long.fromString(value);
        }
        return value.low || value.high ? new protobuf.LongBits(value.low >>> 0, value.high >>> 0) : protobuf.LongBits.zero;
    }

    hash(value) {
        return value ? protobuf.LongBits.from(value).toHash() : '\0\0\0\0\0\0\0\0';
    }
};

protobuf.LongBits.zero = new protobuf.LongBits(0, 0);
protobuf.LongBits.zero.toNumber = function() { return 0; };
protobuf.LongBits.zero.zzDecode = function() { return this; };

protobuf.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Protocol Buffer Error';
        this.message = message;
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Reader = protobuf.Reader;
    module.exports.TextReader = protobuf.TextReader;
    module.exports.Error = protobuf.Error;
    module.exports.Long = protobuf.Long;
    module.exports.get = protobuf.get;
}