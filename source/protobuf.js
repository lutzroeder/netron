
import * as text from './text.js';

const protobuf = {};

protobuf.BinaryReader = class {

    static open(data, offset) {
        offset = offset || 0;
        if (data instanceof Uint8Array) {
            return new protobuf.BufferReader(data, offset);
        }
        if (data.length < 0x20000000) {
            data = data.peek();
            return new protobuf.BufferReader(data, offset);
        }
        return new protobuf.StreamReader(data, offset);
    }

    constructor() {
        this._utf8Decoder = new TextDecoder('utf-8');
    }

    signature() {
        const tags = new Map();
        this._position = 0;
        try {
            if (this._length > 0) {
                const type = this.byte() & 7;
                this.skip(-1);
                if (type !== 4 && type !== 6 && type !== 7) {
                    const length = this.length;
                    while (this._position < length) {
                        const tag = this.uint32();
                        const field = tag >>> 3;
                        const type = tag & 7;
                        if (type > 5 || field === 0) {
                            tags.clear();
                            break;
                        }
                        tags.set(field, type);
                        if (!this._skipType(type)) {
                            tags.clear();
                            break;
                        }
                    }
                }
            }
        } catch {
            tags.clear();
        }
        this._position = 0;
        return tags;
    }

    decode() {
        let tags = {};
        this._position = 0;
        try {
            const decodeMessage = (max) => {
                const length = this._uint32();
                if (length === undefined) {
                    return undefined;
                }
                if (length === 0) {
                    // return 2;
                }
                const end = this.position + length;
                if (end > max) {
                    return undefined;
                }
                try {
                    const tags = {};
                    while (this.position < end) {
                        const tag = this._uint32();
                        if (tag === undefined) {
                            this.seek(end);
                            return 2;
                        }
                        const field = tag >>> 3;
                        const type = tag & 7;
                        if (type > 5 || field === 0) {
                            this.seek(end);
                            return 2;
                        }
                        if (type === 2) {
                            const type = tags[field];
                            if (type !== 2) {
                                const inner = decodeMessage(end);
                                if (this.position > end) {
                                    this.seek(end);
                                    return 2;
                                }
                                if (inner === undefined) {
                                    this.seek(end);
                                    return 2;
                                }
                                if (inner === 2) {
                                    tags[field] = inner;
                                } else if (type) {
                                    for (const [key, value] of Object.entries(inner)) {
                                        if (type[key] === 2 && value !== 2) {
                                            continue;
                                        }
                                        type[key] = value;
                                    }
                                } else {
                                    tags[field] = inner;
                                }
                                continue;
                            }
                        }
                        tags[field] = type;
                        if (!this._skipType(type)) {
                            this.seek(end);
                            return 2;
                        }
                    }
                    if (this.position === end) {
                        return tags;
                    }
                } catch {
                    // continue regardless of error
                }
                this.seek(end);
                return 2;
            };
            if (this._length > 0) {
                const type = this.byte() & 7;
                this.skip(-1);
                if (type !== 4 && type !== 6 && type !== 7) {
                    const length = this.length;
                    while (this.position < length) {
                        const tag = this.uint32();
                        const field = tag >>> 3;
                        const type = tag & 7;
                        if (type > 5 || field === 0) {
                            tags = {};
                            break;
                        }
                        if (type === 2) {
                            const type = tags[field];
                            if (type !== 2) {
                                const inner = decodeMessage(length);
                                if (inner === undefined) {
                                    tags = {};
                                    break;
                                }
                                if (inner === 2) {
                                    tags[field] = inner;
                                } else if (type) {
                                    for (const [name, value] of Object.entries(inner)) {
                                        if (type[name] === 2 && value !== 2) {
                                            continue;
                                        }
                                        type[name] = value;
                                    }
                                } else {
                                    tags[field] = inner;
                                }
                                continue;
                            }
                        }
                        tags[field] = type;
                        if (!this._skipType(type)) {
                            tags = {};
                            break;
                        }
                    }
                }
            }
        } catch {
            tags = {};
        }
        this._position = 0;
        return tags;
    }

    get length() {
        return this._length;
    }

    get position() {
        return this._position;
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
    }

    bytes() {
        const length = this.uint32();
        return this.read(length);
    }

    string() {
        const buffer = this.bytes();
        return this._utf8Decoder.decode(buffer);
    }

    bool() {
        return this.uint32() !== 0;
    }

    uint32() {
        let c = this.byte();
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
        value += (c & 127) * 0x10000000;
        if (c < 128) {
            return value;
        }
        if (this.byte() !== 255 || this.byte() !== 255 || this.byte() !== 255 || this.byte() !== 255 || this.byte() !== 1) {
            throw new protobuf.Error('Varint is not 32-bit.');
        }
        return value;
    }

    _uint32() {
        if (this._position < this._length) {
            let c = this.byte();
            let value = (c & 127) >>> 0;
            if (c < 128) {
                return value;
            }
            if (this._position < this._length) {
                c = this.byte();
                value = (value | (c & 127) << 7) >>> 0;
                if (c < 128) {
                    return value;
                }
                if (this._position < this._length) {
                    c = this.byte();
                    value = (value | (c & 127) << 14) >>> 0;
                    if (c < 128) {
                        return value;
                    }
                    if (this._position < this._length) {
                        c = this.byte();
                        value = (value | (c & 127) << 21) >>> 0;
                        if (c < 128) {
                            return value;
                        }
                        if (this._position < this._length) {
                            c = this.byte();
                            value += (c & 127) * 0x10000000;
                            if (c < 128) {
                                return value;
                            }
                            if (this.byte() !== 255 || this.byte() !== 255 || this.byte() !== 255 || this.byte() !== 255 || this.byte() !== 1) {
                                return undefined;
                            }
                            return value;
                        }
                    }
                }
            }
        }
        return undefined;
    }

    int32() {
        return this.uint32() | 0;
    }

    sint32() {
        const value = this.uint32();
        return value >>> 1 ^ -(value & 1) | 0;
    }

    int64() {
        return BigInt.asIntN(64, this.varint());
    }

    uint64() {
        return this.varint();
    }

    sint64() {
        const value = this.varint();
        return (value >> 1n) ^ (-(value & 1n)); // ZigZag decode
    }

    array(obj, item, tag) {
        if ((tag & 7) === 2) {
            const end = this.uint32() + this._position;
            while (this._position < end) {
                obj.push(item());
            }
        } else {
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
            obj = size > 0x100000 ? new Float32Array(length) : new Array(length);
            for (let i = 0; i < length; i++) {
                obj[i] = this.float();
            }
            this._position = end;
        } else if (obj !== undefined && obj.length <= 0x4000000) {
            obj.push(this.float());
        } else {
            obj = undefined;
            this.float();
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
            for (let i = 0; i < length; i++) {
                obj[i] = this.double();
            }
            this._position = end;
        } else if (obj !== undefined && obj.length < 1000000) {
            obj.push(this.double());
        } else {
            obj = undefined;
            this.double();
        }
        return obj;
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._length) {
            this._unexpected();
        }
    }

    skipType(type) {
        switch (type) {
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
                while ((type = this.uint32() & 7) !== 4) {
                    this.skipType(type);
                }
                break;
            case 5:
                this.skip(4);
                break;
            default:
                throw new protobuf.Error(`Invalid type '${type}' at offset ${this._position}.`);
        }
    }

    _skipType(type) {
        switch (type) {
            case 0: {
                // const max = this._position + 9;
                do {
                    if (this._position >= this._length /* || this._position > max */) {
                        return false;
                    }
                }
                while (this.byte() & 128);
                break;
            }
            case 1: {
                const position = this._position + 8;
                if (position > this._length) {
                    return false;
                }
                this._position = position;
                break;
            }
            case 2: {
                const length = this.uint32();
                if (length === undefined) {
                    return false;
                }
                const position = this._position + length;
                if (position > this._end) {
                    return false;
                }
                this._position = position;
                break;
            }
            case 3: {
                for (;;) {
                    const tag = this.uint32();
                    if (tag === undefined) {
                        return false;
                    }
                    const wireType = tag & 7;
                    if (wireType === 4) {
                        break;
                    }
                    if (!this._skipType(wireType)) {
                        return false;
                    }
                }
                break;
            }
            case 5: {
                const position = this._position + 4;
                if (position > this._length) {
                    return false;
                }
                this._position = position;
                break;
            }
            default: {
                return false;
            }
        }
        return true;
    }

    entry(obj, key, value) {
        this.skipVarint();
        this.skip(1);
        let k = key();
        if (!Number.isInteger(k) && typeof k !== 'string') {
            k = Number(k);
        }
        this.skip(1);
        const v = value();
        obj[k] = v;
    }

    _unexpected() {
        throw new RangeError('Unexpected end of file.');
    }
};

protobuf.BufferReader = class extends protobuf.BinaryReader {

    constructor(buffer, offset = 0) {
        super();
        this._buffer = buffer;
        this._length = buffer.length;
        this._position = offset;
        this._view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }

    skipVarint() {
        do {
            if (this._position >= this._length) {
                this._unexpected();
            }
        }
        while (this._buffer[this._position++] & 128);
    }

    read(length) {
        const position = this._position;
        this.skip(length);
        return this._buffer.slice(position, this._position);
    }

    byte() {
        if (this._position < this._length) {
            return this._buffer[this._position++];
        }
        throw new RangeError('Unexpected end of file.');
    }

    fixed64() {
        const position = this._position;
        this.skip(8);
        return this._view.getBigUint64(position, true);
    }

    sfixed64() {
        const position = this._position;
        this.skip(8);
        return this._view.getBigInt64(position, true);
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

    varint() {
        let value = 0n;
        let shift = 0n;
        for (let i = 0; i < 10 && this._position < this._length; i++) {
            const byte = this._buffer[this._position++];
            value |= BigInt(byte & 0x7F) << shift;
            if ((byte & 0x80) === 0) {
                return value;
            }
            shift += 7n;
        }
        throw new protobuf.Error('Invalid varint value.');
    }
};

protobuf.StreamReader = class extends protobuf.BinaryReader {

    constructor(stream, offset) {
        super(new Uint8Array(0));
        this._stream = stream;
        this._length = stream.length;
        this.seek(offset || 0);
    }

    skipVarint() {
        do {
            if (this._position >= this._length) {
                this._unexpected();
            }
        }
        while (this.byte() & 128);
    }

    read(length) {
        this._stream.seek(this._position);
        this.skip(length);
        return this._stream.read(length);
    }

    byte() {
        const position = this._fill(1);
        return this._view.getUint8(position);
    }

    fixed64() {
        const position = this._fill(8);
        return this._view.getBigUint64(position, true);
    }

    sfixed64() {
        const position = this._fill(8);
        return this._view.getBigInt64(position, true);
    }

    fixed32() {
        const position = this._fill(4);
        return this._view.getUint32(position, true);
    }

    sfixed32() {
        const position = this._fill(4);
        return this._view.getInt32(position, true);
    }

    float() {
        const position = this._fill(4);
        return this._view.getFloat32(position, true);
    }

    double() {
        const position = this._fill(8);
        return this._view.getFloat64(position, true);
    }

    varint() {
        let value = 0n;
        let shift = 0n;
        for (let i = 0; i < 10 && this._position < this._length; i++) {
            const byte = this.byte();
            value |= BigInt(byte & 0x7F) << shift;
            if ((byte & 0x80) === 0) {
                return value;
            }
            shift += 7n;
        }
        throw new protobuf.Error('Invalid varint value.');
    }

    _fill(length) {
        if (this._position + length > this._length) {
            throw new Error(`Expected ${this._position + length - this._length} more bytes. The file might be corrupted. Unexpected end of file.`);
        }
        if (!this._buffer || this._position < this._offset || this._position + length > this._offset + this._buffer.length) {
            this._offset = this._position;
            this._stream.seek(this._offset);
            const size = Math.min(0x10000000, this._length - this._offset);
            this._buffer = this._stream.read(size);
            this._view = new DataView(this._buffer.buffer, this._buffer.byteOffset, this._buffer.byteLength);
        }
        const position = this._position;
        this._position += length;
        return position - this._offset;
    }
};

protobuf.TextReader = class {

    static open(data) {
        if (data) {
            const buffer = data instanceof Uint8Array ? data : data.peek();
            const decoder = text.Decoder.open(buffer);
            let first = true;
            for (let i = 0; i < 0x100; i++) {
                const c = decoder.decode();
                if (c === undefined) {
                    if (i === 0) {
                        return null;
                    }
                    break;
                }
                if (c === '\0') {
                    return null;
                }
                const whitespace = c === ' ' || c === '\n' || c === '\r' || c === '\t';
                if (c < ' ' && !whitespace) {
                    return null;
                }
                if (first && !whitespace) {
                    first = false;
                    if (c === '#') {
                        let c = '';
                        do {
                            c = decoder.decode();
                        }
                        while (c !== undefined && c !== '\n');
                        if (c === undefined) {
                            break;
                        }
                        continue;
                    }
                    if (c === '[') {
                        continue;
                    }
                    if (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z') {
                        continue;
                    }
                    return null;
                }
            }
            return new protobuf.TextReader(buffer);
        }
        return null;
    }

    constructor(buffer) {
        this._decoder = text.Decoder.open(buffer);
        this.reset();
    }

    signature() {
        const tags = new Map();
        this.reset();
        try {
            this.start(false);
            while (!this.end()) {
                const tag = this.tag();
                if (this.token() === '{') {
                    this.start();
                    tags.set(tag, true);
                    while (!this.end()) {
                        const subtag = this.tag();
                        tags.set(`${tag}.${subtag}`, true);
                        this.skip();
                        this.match(',');
                    }
                } else {
                    this.skip();
                    tags.set(tag, true);
                }
            }
        } catch {
            if (tags.has('[')) {
                tags.clear();
            }
        }
        this.reset();
        return tags;
    }

    reset() {
        this._decoder.position = 0;
        this._position = 0;
        this._depth = 0;
        this._arrayDepth = 0;
        this._token = '';
        this.next();
    }

    start() {
        if (this._depth > 0) {
            this.expect('{');
        }
        this._depth++;
    }

    end() {
        if (this._depth <= 0) {
            throw new protobuf.Error(`Invalid depth ${this.location()}`);
        }
        if (this._token === '}') {
            this.expect('}');
            this.match(';');
            this._depth--;
            return true;
        }
        if (this._token === undefined) {
            if (this._depth !== 1) {
                throw new protobuf.Error(`Unexpected end of input ${this.location()}`);
            }
            this._depth--;
            return true;
        }
        return false;
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
            throw new protobuf.Error(`Couldn't parse integer '${token}' ${this.location()}`);
        }
        this.next();
        this.semicolon();
        return value;
    }

    double() {
        let value = 0;
        let token = this._token;
        switch (token) {
            case 'nan': value = NaN; break;
            case 'inf': value = Infinity; break;
            case '-inf': value = -Infinity; break;
            default:
                if (token.endsWith('f')) {
                    token = token.substring(0, token.length - 1);
                }
                value = Number.parseFloat(token);
                if (Number.isNaN(token - value)) {
                    throw new protobuf.Error(`Couldn't parse float '${token}' ${this.location()}`);
                }
                break;
        }
        this.next();
        this.semicolon();
        return value;
    }

    float() {
        return this.double();
    }

    uint32() {
        return this.integer();
    }

    int32() {
        return this.integer();
    }

    sint32() {
        return this.integer();
    }

    int64() {
        return BigInt(this.integer());
    }

    uint64() {
        return BigInt.asUintN(64, BigInt(this.integer()));
    }

    sint64() {
        return BigInt(this.integer());
    }

    fixed64() {
        return BigInt.asUintN(64, BigInt(this.integer()));
    }

    sfixed64() {
        return BigInt(this.integer());
    }

    fixed32() {
        return this.integer();
    }

    sfixed32() {
        return this.integer();
    }

    string() {
        const token = this._token;
        if (token.length < 2) {
            throw new protobuf.Error(`String is too short ${this.location()}`);
        }
        const quote = token.substring(0, 1);
        if (quote !== "'" && quote !== '"') {
            throw new protobuf.Error(`String is not in quotes ${this.location()}`);
        }
        if (quote !== token.substring(token.length - 1)) {
            throw new protobuf.Error(`String quotes do not match ${this.location()}`);
        }
        const value = token.substring(1, token.length - 1);
        this.next();
        this.semicolon();
        return value;
    }

    bool() {
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
            default:
                throw new protobuf.Error(`Couldn't parse boolean '${token}' ${this.location()}`);
        }
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
        let value = 0;
        if (Object.prototype.hasOwnProperty.call(type, token)) {
            value = type[token];
        } else {
            value = Number.parseInt(token, 10);
            if (Number.isNaN(token - value)) {
                throw new protobuf.Error(`Couldn't parse enum '${token === undefined ? '' : token}' ${this.location()}`);
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
        } else {
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
        } else {
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
        let k = '';
        let v = null;
        while (!this.end()) {
            const tag = this.tag();
            switch (tag) {
                case 'key':
                    k = key();
                    break;
                case 'value':
                    v = value();
                    break;
                default:
                    throw new protobuf.Error(`Unsupported entry tag '${tag}'.`);
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
        } else {
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
        let content = '';
        while (this._decoder.position < end) {
            content += this._decoder.decode();
        }
        this._decoder.position = position;
        return content;
    }

    skip() {
        switch (this._token) {
            case '{': {
                const depth = this._depth;
                this.start();
                while (!this.end() || depth < this._depth) {
                    if (this._token === '{') {
                        this.start();
                    } else if (this._token !== '}') {
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
                    } else if (this._token === undefined) {
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
        throw new protobuf.Error(`Unexpected token '${token}' ${this.location()}`);
    }

    field(token /*, module */) {
        throw new protobuf.Error(`Unsupported field '${token}' ${this.location()}`);
    }

    token() {
        return this._token;
    }

    next() {
        if (this._token === undefined) {
            throw new protobuf.Error(`Unexpected end of input ${this.location()}`);
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
                            this._token = undefined;
                            return;
                        }
                    }
                    while (c !== '\n');
                    this._position = this._decoder.position;
                    c = this._decoder.decode();
                    continue;
                default:
                    break;
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
                let content = c;
                for (;;) {
                    c = this._decoder.decode();
                    if (c === undefined || c === '\n') {
                        throw new protobuf.Error(`Unexpected end of string ${this.location()}`);
                    }
                    if (c === '\\') {
                        c = this._decoder.decode();
                        if (c === undefined || c === '\n') {
                            throw new protobuf.Error(`Unexpected end of string ${this.location()}`);
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
                                    let c = this._decoder.decode();
                                    if (c === undefined) {
                                        throw new protobuf.Error(`Unexpected end of string ${this.location()}`);
                                    }
                                    c = c.charCodeAt(0);
                                    if (c >= 65 && c <= 70) {
                                        c -= 55;
                                    } else if (c >= 97 && c <= 102) {
                                        c -= 87;
                                    } else if (c >= 48 && c <= 57) {
                                        c -= 48;
                                    } else {
                                        c = -1;
                                    }
                                    if (c === -1) {
                                        throw new protobuf.Error(`Unexpected hex digit '${c}' in bytes string ${this.location()}`);
                                    }
                                    value = value << 4 | c;
                                }
                                c = String.fromCharCode(value);
                                break;
                            }
                            default: {
                                if (c < '0' || c > '9') {
                                    throw new protobuf.Error(`Unexpected character '${c}' in string ${this.location()}`);
                                }
                                let value = 0;
                                let od = c;
                                if (od < '0' || od > '9') {
                                    throw new protobuf.Error(`Unexpected octal digit '${od}' in bytes string ${this.location()}`);
                                }
                                od = od.charCodeAt(0);
                                value = value << 3 | od - 48;
                                od = this._decoder.decode();
                                if (od === undefined) {
                                    throw new protobuf.Error(`Unexpected end of string ${this.location()}`);
                                }
                                if (od < '0' || od > '9') {
                                    throw new protobuf.Error(`Unexpected octal digit '${od}' in bytes string ${this.location()}`);
                                }
                                od = od.charCodeAt(0);
                                value = value << 3 | od - 48;
                                od = this._decoder.decode();
                                if (od === undefined) {
                                    throw new protobuf.Error(`Unexpected end of string ${this.location()}`);
                                }
                                if (od < '0' || od > '9') {
                                    throw new protobuf.Error(`Unexpected octal digit '${od}' in bytes string ${this.location()}`);
                                }
                                od = od.charCodeAt(0);
                                value = value << 3 | od - 48;
                                c = String.fromCharCode(value);
                                break;
                            }
                        }
                        content += c;
                        continue;
                    } else {
                        content += c;
                        if (c === quote) {
                            break;
                        }
                    }
                }
                this._token = content;
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
                if (token === '-' && c === 'i' && this._decoder.decode() === 'n' && this._decoder.decode() === 'f') {
                    token = '-inf';
                    position = this._decoder.position;
                }
                if (token === '-' || token === '+' || token === '.') {
                    throw new protobuf.Error(`Unexpected token '${token}' ${this.location()}`);
                }
                this._decoder.position = position;
                this._token = token;
                return;
            }
            default: {
                throw new protobuf.Error(`Unexpected token '${c}' ${this.location()}`);
            }
        }
    }

    expect(value) {
        if (this._token !== value) {
            throw new protobuf.Error(`Unexpected '${this._token}' instead of '${value}' ${this.location()}`);
        }
        this.next();
    }

    match(value) {
        if (value === this._token) {
            this.next();
            return true;
        }
        return false;
    }

    location() {
        let line = 1;
        let column = 1;
        this._decoder.position = 0;
        let c = '';
        do {
            if (this._decoder.position === this._position) {
                return `at ${line}:${column}.`;
            }
            c = this._decoder.decode();
            if (c === '\n') {
                line++;
                column = 1;
            } else {
                column++;
            }
        }
        while (c !== undefined);
        return `at ${line}:${column}.`;
    }

    semicolon() {
        if (this._arrayDepth === 0) {
            this.match(';');
        }
    }
};

protobuf.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Protocol Buffers Error';
        this.message = message;
    }
};

export const BinaryReader = protobuf.BinaryReader;
export const TextReader = protobuf.TextReader;
