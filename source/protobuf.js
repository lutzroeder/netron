
/* jshint esversion: 6 */

var protobuf = protobuf || {};
var base = base || require('./base');

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
        this._view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._utf8Decoder = new TextDecoder('utf-8');
    }

    static create(buffer) {
        return new protobuf.Reader(buffer);
    }

    get length() {
        return this._length;
    }

    get position() {
        return this._position;
    }

    string() {
        return this._utf8Decoder.decode(this.bytes());
    }

    bool() {
        return this.uint32() !== 0;
    }

    byte() {
        if (this._position < this._length) {
            return this._buffer[this._position++];
        }
        throw new RangeError('Unexpected end of file.');
    }

    bytes() {
        const length = this.uint32();
        const position = this._position;
        this.skip(length);
        return this._buffer.slice(position, this._position);
    }

    uint32() {
        let c;
        c = this.byte();
        let value = (c & 127) >>> 0;
        if (c < 128) {
            return value;
        }
        c = this.byte();
        value = (value | (c & 127) <<  7) >>> 0;
        if (c < 128) {
            return value;
        }
        c = this.byte();
        value = (value | (c & 127) << 14) >>> 0;
        if (c < 128) {
            return value;
        }
        c = this.byte();
        value = (value | (c & 127) << 21) >>> 0;
        if (c < 128) {
            return value;
        }
        c = this.byte();
        value = (value | (c & 15) << 28) >>> 0;
        if (c < 128) {
            return value;
        }
        if (this.byte() !== 255 || this.byte() !== 255 || this.byte() !== 255 || this.byte() !== 255 || this.byte() !== 1) {
            throw new protobuf.Error('Varint is not 32-bit.');
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
        return this._varint().toInt64();
    }

    uint64() {
        return this._varint().toInt64();
    }

    sint64() {
        return this._varint().zzDecode().toInt64();
    }

    fixed64() {
        const position = this._position;
        this.skip(8);
        return this._view.getUint64(position, true);
    }

    sfixed64() {
        const position = this._position;
        this.skip(8);
        return this._view.getInt64(position, true);
    }

    fixed32() {
        const position = this._position;
        this.skip(4);
        return this._view.getUint32(position, true);
    }

    sfixed32() {
        const position = this._position;
        this.skip(4);
        return this._view.getInt32(position, true);
    }

    float() {
        const position = this._position;
        this.skip(4);
        return this._view.getFloat32(position, true);
    }

    double() {
        const position = this._position;
        this.skip(8);
        return this._view.getFloat64(position, true);
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
            if (end > this._length) {
                this._unexpected();
            }
            const length = size >>> 2;
            obj = size > 1048576 ? new Float32Array(length) : new Array(length);
            let position = this._position;
            for (let i = 0; i < length; i++) {
                obj[i] = this._view.getFloat32(position, true);
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
            if (end > this._length) {
                this._unexpected();
            }
            const length = size >>> 3;
            obj = size > 1048576 ? new Float64Array(length) : new Array(length);
            let position = this._position;
            for (let i = 0; i < length; i++) {
                obj[i] = this._view.getFloat64(position, true);
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

    skip(offset) {
        this._position += offset;
        if (this._position > this._length) {
            this._unexpected();
        }
    }

    skipVarint() {
        do {
            if (this._position >= this._length) {
                this._unexpected();
            }
        }
        while (this._buffer[this._position++] & 128);
    }

    skipType(wireType) {
        switch (wireType) {
            case 0:
                this.skipVarint();
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
                throw new protobuf.Error('Invalid type ' + wireType + ' at offset ' + this._position + '.');
        }
    }

    entry(obj, key, value) {
        this.skipVarint();
        this._position++;
        let k = key();
        if (!Number.isInteger(k) && typeof k !== 'string') {
            k = k.toNumber();
        }
        this._position++;
        const v = value();
        obj[k] = v;
    }

    _varint() {
        const bits = new protobuf.LongBits(0, 0);
        let i = 0;
        if (this._length - this._position > 4) { // fast route (lo)
            for (; i < 4; ++i) {
                // 1st..4th
                bits.lo = (bits.lo | (this._buffer[this._position] & 127) << i * 7) >>> 0;
                if (this._buffer[this._position++] < 128) {
                    return bits;
                }
            }
            // 5th
            bits.lo = (bits.lo | (this._buffer[this._position] & 127) << 28) >>> 0;
            bits.hi = (bits.hi | (this._buffer[this._position] & 127) >>  4) >>> 0;
            if (this._buffer[this._position++] < 128) {
                return bits;
            }
            i = 0;
        }
        else {
            for (; i < 3; i++) {
                if (this._position >= this._length) {
                    this._unexpected();
                }
                bits.lo = (bits.lo | (this._buffer[this._position] & 127) << i * 7) >>> 0;
                if (this._buffer[this._position++] < 128) {
                    return bits;
                }
            }
            bits.lo = (bits.lo | (this._buffer[this._position++] & 127) << i * 7) >>> 0;
            return bits;
        }
        if (this._length - this._position > 4) {
            for (; i < 5; ++i) {
                bits.hi = (bits.hi | (this._buffer[this._position] & 127) << i * 7 + 3) >>> 0;
                if (this._buffer[this._position++] < 128) {
                    return bits;
                }
            }
        }
        else {
            for (; i < 5; ++i) {
                if (this._position >= this._length) {
                    this._unexpected();
                }
                bits.hi = (bits.hi | (this._buffer[this._position] & 127) << i * 7 + 3) >>> 0;
                if (this._buffer[this._position++] < 128) {
                    return bits;
                }
            }
        }
        throw new protobuf.Error('Invalid varint encoding.');
    }

    _unexpected() {
        throw new RangeError('Unexpected end of file.');
    }
};

protobuf.TextReader = class {

    constructor(buffer) {
        this._decoder = base.TextDecoder.create(buffer);
        this._position = 0;
        this._token = undefined;
        this._depth = 0;
        this._arrayDepth = 0;
        this._token = '';
        this.next();
    }

    static create(buffer) {
        return new protobuf.TextReader(buffer);
    }

    start() {
        if (this._depth > 0) {
            this.expect('{');
        }
        this._depth++;
    }

    end() {
        if (this._depth > 0 && this._token === '}') {
            this.expect('}');
            this.match(';');
            this._depth--;
            return true;
        }
        return this._token === undefined;
    }

    tag() {
        const name = this._token;
        this.next();
        if (this._token !== '[' && this._token !== '{') {
            this.expect(':');
        }
        return name;
    }

    integer() {
        const token = this._token;
        const value = Number.parseInt(token, 10);
        if (Number.isNaN(token - value)) {
            throw new protobuf.Error("Couldn't parse integer '" + token + "'" + this.location());
        }
        this.next();
        this.semicolon();
        return value;
    }

    float() {
        let token = this._token;
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
        this.next();
        this.semicolon();
        return value;
    }

    string() {
        const token = this._token;
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
        this.next();
        this.semicolon();
        return value;
    }

    boolean() {
        const token = this._token;
        switch (token) {
            case 'true':
            case 'True':
            case '1':
                this.next();
                this.semicolon();
                return true;
            case 'false':
            case 'False':
            case '0':
                this.next();
                this.semicolon();
                return false;
        }
        throw new protobuf.Error("Couldn't parse boolean '" + token + "'" + this.location());
    }

    bytes() {
        const token = this.string();
        const length = token.length;
        const array = new Uint8Array(length);
        for (let i = 0; i < length; i++) {
            array[i] = token.charCodeAt(i);
        }
        return array;
    }

    enum(type) {
        const token = this._token;
        let value;
        if (Object.prototype.hasOwnProperty.call(type, token)) {
            value = type[token];
        }
        else {
            value = Number.parseInt(token, 10);
            if (Number.isNaN(token - value)) {
                throw new protobuf.Error("Couldn't parse enum '" + token + "'" + this.location());
            }
        }
        this.next();
        this.semicolon();
        return value;
    }

    any(type) {
        this.start();
        const message = type();
        if (this._token.startsWith('[') && this._token.endsWith(']')) {
            message.type_url = this._token.substring(1, this._token.length - 1).trim();
            this.next();
            this.match(':');
            message.value = this.read();
            this.match(';');
            if (!this.end()) {
                this.expect('}');
            }
        }
        else {
            while (!this.end()) {
                const tag = this.tag();
                switch (tag) {
                    case "type_url":
                        message.type_url = this.string();
                        break;
                    case "value":
                        message.value = this.bytes();
                        break;
                    default:
                        this.field(tag, message);
                        break;
                }
            }
        }
        return message;
    }

    anyarray(obj, type) {
        this.start();
        if (this._token.startsWith('[') && this._token.endsWith(']')) {
            while (!this.end()) {
                if (this._token.startsWith('[') && this._token.endsWith(']')) {
                    const message = type();
                    message.type_url = this._token.substring(1, this._token.length - 1).trim();
                    this.next();
                    this.match(':');
                    message.value = this.read();
                    this.match(';');
                    obj.push(message);
                    continue;
                }
                this.expect('[');
            }
        }
        else {
            const message = type();
            while (!this.end()) {
                const tag = this.tag();
                switch (tag) {
                    case "type_url":
                        message.type_url = this.string();
                        break;
                    case "value":
                        message.value = this.bytes();
                        break;
                    default:
                        this.field(tag, message);
                        break;
                }
            }
            obj.push(message);
        }
    }

    entry(obj, key, value) {
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
                switch (this._token) {
                    case ',':
                        this.next();
                        break;
                    case ']':
                        break;
                    default:
                        this.handle(this._token);
                        break;
                }
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

    read() {
        const start = this._position;
        this.skip();
        const end = this._position;
        const position = this._decoder.position;
        this._decoder.position = start;
        let text = '';
        while (this._decoder.position < end) {
            text += this._decoder.decode();
        }
        this._decoder.position = position;
        return text;
    }

    skip() {
        switch (this._token) {
            case '{': {
                const depth = this._depth;
                this.start();
                while (!this.end() || depth < this._depth) {
                    if (this._token === '{') {
                        this.start();
                    }
                    else if (this._token !== '}') {
                        this.next();
                        this.match(';');
                    }
                }
                break;
            }
            case '[': {
                const depth = this._arrayDepth;
                this.first();
                while (!this.last() || depth < this._arrayDepth) {
                    this.next();
                    if (this._token === '[') {
                        this.first();
                    }
                    else if (this._token === undefined) {
                        this.handle(this._token);
                    }
                }
                break;
            }
            default: {
                this.next();
                this.semicolon();
                break;
            }
        }
    }

    handle(token) {
        throw new protobuf.Error("Unexpected token '" + token + "'" + this.location());
    }

    field(token /*, module */) {
        throw new protobuf.Error("Unknown field '" + token + "'" + this.location());
    }

    token() {
        return this._token;
    }

    next() {
        if (this._token === undefined) {
            throw new protobuf.Error('Unexpected end of input' + this.location());
        }
        this._position = this._decoder.position;
        let c = this._decoder.decode();
        for (;;) {
            switch (c) {
                case ' ':
                case '\n':
                case '\r':
                case '\t':
                    this._position = this._decoder.position;
                    c = this._decoder.decode();
                    continue;
                case '#':
                    do {
                        c = this._decoder.decode();
                        if (c === undefined) {
                            this._token === undefined;
                            return;
                        }
                    }
                    while (c !== '\n');
                    this._position = this._decoder.position;
                    c = this._decoder.decode();
                    continue;
            }
            break;
        }
        if (c === undefined) {
            this._token = undefined;
            return;
        }
        if (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c === '_' || c === '$') {
            let token = c;
            let position = this._decoder.position;
            for (;;) {
                c = this._decoder.decode();
                if (c === undefined || c === '\n') {
                    break;
                }
                if (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c === '_' || c === '+' || c === '-') {
                    token += c;
                    position = this._decoder.position;
                    continue;
                }
                break;
            }
            this._decoder.position = position;
            this._token = token;
            return;
        }
        switch (c) {
            case '{':
            case '}':
            case ':':
            case ',':
            case ']':
            case ';':
                this._token = c;
                return;
            case '[': {
                let token = c;
                let position = this._decoder.position;
                let x = this._decoder.decode();
                if ((x !== undefined) && x >= 'a' && x <= 'z' || x >= 'A' && x <= 'Z') {
                    token += x;
                    for (;;) {
                        x = this._decoder.decode();
                        if (x === undefined || x === '\n') {
                            break;
                        }
                        if (x >= 'a' && x <= 'z' || x >= 'A' && x <= 'Z' || x >= '0' && x <= '9' || x === '.' || x === '/') {
                            token += x;
                            position = this._decoder.position;
                            continue;
                        }
                        if (x === ']') {
                            this._token = token + x;
                            return;
                        }
                    }
                }
                this._decoder.position = position;
                this._token = '[';
                return;
            }
            case '"':
            case "'": {
                const quote = c;
                let text = c;
                for (;;) {
                    c = this._decoder.decode();
                    if (c === undefined || c === '\n') {
                        throw new protobuf.Error('Unexpected end of string' + this.location());
                    }
                    if (c == '\\') {
                        c = this._decoder.decode();
                        if (c === undefined || c === '\n') {
                            throw new protobuf.Error('Unexpected end of string' + this.location());
                        }
                        switch (c) {
                            case '\\': c = '\\'; break;
                            case "'": c = "'"; break;
                            case '"': c = '"'; break;
                            case 'r': c = '\r'; break;
                            case 'n': c = '\n'; break;
                            case 't': c = '\t'; break;
                            case 'b': c = '\b'; break;
                            case 'x':
                            case 'X': {
                                let value = 0;
                                for (let xi = 0; xi < 2; xi++) {
                                    let xd = this._decoder.decode();
                                    if (xd === undefined) {
                                        throw new protobuf.Error('Unexpected end of string' + this.location());
                                    }
                                    xd = xd.charCodeAt(0);
                                    xd = xd >= 65 && xd <= 70 ? xd - 55 : xd >= 97 && xd <= 102 ? xd - 87 : xd >= 48 && xd <= 57 ? xd - 48 : -1;
                                    if (xd === -1) {
                                        throw new protobuf.Error("Unexpected hex digit '" + xd + "' in bytes string" + this.location());
                                    }
                                    value = value << 4 | xd;
                                }
                                c = String.fromCharCode(value);
                                break;
                            }
                            default: {
                                if (c < '0' || c > '9') {
                                    throw new protobuf.Error("Unexpected character '" + c + "' in string" + this.location());
                                }
                                let value = 0;
                                let od = c;
                                if (od < '0' || od > '9') {
                                    throw new protobuf.Error("Unexpected octal digit '" + od + "' in bytes string" + this.location());
                                }
                                od = od.charCodeAt(0);
                                value = value << 3 | od - 48;
                                od = this._decoder.decode();
                                if (od === undefined) {
                                    throw new protobuf.Error('Unexpected end of string' + this.location());
                                }
                                if (od < '0' || od > '9') {
                                    throw new protobuf.Error("Unexpected octal digit '" + od + "' in bytes string" + this.location());
                                }
                                od = od.charCodeAt(0);
                                value = value << 3 | od - 48;
                                od = this._decoder.decode();
                                if (od === undefined) {
                                    throw new protobuf.Error('Unexpected end of string' + this.location());
                                }
                                if (od < '0' || od > '9') {
                                    throw new protobuf.Error("Unexpected octal digit '" + od + "' in bytes string" + this.location());
                                }
                                od = od.charCodeAt(0);
                                value = value << 3 | od - 48;
                                c = String.fromCharCode(value);
                                break;
                            }
                        }
                        text += c;
                        continue;
                    }
                    else {
                        text += c;
                        if (c === quote) {
                            break;
                        }
                    }
                }
                this._token = text;
                return;
            }
            case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
            case '-': case '+': case '.': {
                let token = c;
                let position = this._decoder.position;
                for (;;) {
                    c = this._decoder.decode();
                    if (c === undefined || c === '\n') {
                        break;
                    }
                    if ((c >= '0' && c <= '9') || c === '_' || c === '+' || c === '-' || c === '.' || c === 'e' || c === 'E') {
                        token += c;
                        position = this._decoder.position;
                        continue;
                    }
                    break;
                }
                this._decoder.position = position;
                this._token = token;
                return;
            }
        }
        throw new protobuf.Error("Unexpected token '" + c + "'" + this.location());
    }

    expect(value) {
        if (this._token !== value) {
            throw new protobuf.Error("Unexpected '" + this._token + "' instead of '" + value + "'" + this.location());
        }
        this.next();
    }

    match(value) {
        if (value == this._token) {
            this.next();
            return true;
        }
        return false;
    }

    location() {
        let line = 1;
        let column = 1;
        this._decoder.position = 0;
        let c;
        do {
            if (this._decoder.position === this._position) {
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

    semicolon() {
        if (this._arrayDepth === 0) {
            this.match(';');
        }
    }
};

protobuf.Int64 = base.Int64;
protobuf.Uint64 = base.Uint64;

protobuf.LongBits = class {

    constructor(lo, hi) {
        this.lo = lo >>> 0;
        this.hi = hi >>> 0;
    }

    zzDecode() {
        const mask = -(this.lo & 1);
        this.lo  = ((this.lo >>> 1 | this.hi << 31) ^ mask) >>> 0;
        this.hi  = ( this.hi >>> 1                  ^ mask) >>> 0;
        return this;
    }

    toUint64() {
        return new base.Uint64(this.lo, this.hi);
    }

    toInt64() {
        return new base.Int64(this.lo, this.hi);
    }
};

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
    module.exports.Int64 = protobuf.Int64;
    module.exports.Uint64 = protobuf.Uint64;
    module.exports.get = protobuf.get;
}