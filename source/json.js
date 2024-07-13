
import * as text from './text.js';

const json = {};
const bson = {};

json.TextReader = class {

    static open(data) {
        const decoder = text.Decoder.open(data);
        let state = '';
        for (let i = 0; i < 0x1000; i++) {
            const c = decoder.decode();
            if (state === 'match') {
                break;
            }
            if (c === undefined || c === '\0') {
                if (state === '') {
                    return null;
                }
                break;
            }
            if (c <= ' ') {
                if (c !== ' ' && c !== '\n' && c !== '\r' && c !== '\t') {
                    return null;
                }
                continue;
            }
            switch (state) {
                case '':
                    if (c === '{') {
                        state = '{}';
                    } else if (c === '[') {
                        state = '[]';
                    } else {
                        return null;
                    }
                    break;
                case '[]':
                    if (c !== '"' && c !== '-' && c !== '+' && c !== '{' && c !== '[' && (c < '0' || c > '9')) {
                        return null;
                    }
                    state = 'match';
                    break;
                case '{}':
                    if (c !== '"') {
                        return null;
                    }
                    state = 'match';
                    break;
                default:
                    break;
            }
        }
        return new json.TextReader(decoder);
    }

    constructor(decoder) {
        this._decoder = decoder;
        this._decoder.position = 0;
        this._escape = { '"': '"', '\\': '\\', '/': '/', b: '\b', f: '\f', n: '\n', r: '\r', t: '\t' };
    }

    read() {
        const stack = [];
        this._decoder.position = 0;
        this._position = 0;
        this._char = this._decoder.decode();
        this._whitespace();
        let obj = null;
        let first = true;
        for (;;) {
            if (Array.isArray(obj)) {
                this._whitespace();
                let c = this._char;
                if (c === ']') {
                    this._next();
                    this._whitespace();
                    if (stack.length > 0) {
                        obj = stack.pop();
                        first = false;
                        continue;
                    }
                    if (this._char !== undefined) {
                        this._unexpected();
                    }
                    return obj;
                }
                if (!first) {
                    if (this._char !== ',') {
                        this._unexpected();
                    }
                    this._next();
                    this._whitespace();
                    c = this._char;
                }
                first = false;
                switch (c) {
                    case '{': {
                        this._next();
                        stack.push(obj);
                        const item = {};
                        obj.push(item);
                        obj = item;
                        first = true;
                        break;
                    }
                    case '[': {
                        this._next();
                        stack.push(obj);
                        const item = [];
                        obj.push(item);
                        obj = item;
                        first = true;
                        break;
                    }
                    default: {
                        obj.push(c === '"' ? this._string() : this._literal());
                        break;
                    }
                }
            } else if (obj instanceof Object) {
                this._whitespace();
                let c = this._char;
                if (c === '}') {
                    this._next();
                    this._whitespace();
                    if (stack.length > 0) {
                        obj = stack.pop();
                        first = false;
                        continue;
                    }
                    if (this._char !== undefined) {
                        this._unexpected();
                    }
                    return obj;
                }
                if (!first) {
                    if (this._char !== ',') {
                        this._unexpected();
                    }
                    this._next();
                    this._whitespace();
                    c = this._char;
                }
                first = false;
                if (c === '"') {
                    const key = this._string();
                    switch (key) {
                        case '__proto__':
                        case 'constructor':
                            throw new json.Error(`Invalid key '${key}' ${this._location()}`);
                        default:
                            break;
                    }
                    this._whitespace();
                    if (this._char !== ':') {
                        this._unexpected();
                    }
                    this._next();
                    this._whitespace();
                    c = this._char;
                    switch (c) {
                        case '{': {
                            this._next();
                            stack.push(obj);
                            const value = {};
                            obj[key] = value;
                            obj = value;
                            first = true;
                            break;
                        }
                        case '[': {
                            this._next();
                            stack.push(obj);
                            const value = [];
                            obj[key] = value;
                            obj = value;
                            first = true;
                            break;
                        }
                        default: {
                            obj[key] = c === '"' ? this._string() : this._literal();
                            break;
                        }
                    }
                    this._whitespace();
                    continue;
                }
                this._unexpected();
            } else {
                const c = this._char;
                switch (c) {
                    case '{': {
                        this._next();
                        this._whitespace();
                        obj = {};
                        first = true;
                        break;
                    }
                    case '[': {
                        this._next();
                        this._whitespace();
                        obj = [];
                        first = true;
                        break;
                    }
                    default: {
                        let value = null;
                        if (c === '"') {
                            value = this._string();
                        } else if (c >= '0' && c <= '9') {
                            value = this._number();
                        } else {
                            value = this._literal();
                        }
                        this._whitespace();
                        if (this._char !== undefined) {
                            this._unexpected();
                        }
                        return value;
                    }
                }
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
            case 't': this._expect('true'); return true;
            case 'f': this._expect('false'); return false;
            case 'n': this._expect('null'); return null;
            case 'N': this._expect('NaN'); return NaN;
            case 'I': this._expect('Infinity'); return Infinity;
            case '-': return this._number();
            default: this._unexpected();
        }
        return null;
    }

    _number() {
        let value = '';
        if (this._char === '-') {
            value = '-';
            this._next();
        }
        if (this._char === 'I') {
            this._expect('Infinity');
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
        return Number(value);
    }

    _string() {
        let value = '';
        this._next();
        while (this._char !== '"') {
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
                } else if (this._escape[this._char]) {
                    value += this._escape[this._char];
                    this._next();
                } else {
                    this._unexpected();
                }
            } else if (this._char < ' ') {
                this._unexpected();
            } else {
                value += this._char;
                this._next();
            }
        }
        this._next();
        return value;
    }

    _expect(value) {
        for (let i = 0; i < value.length; i++) {
            if (value[i] !== this._char) {
                this._unexpected();
            }
            this._next();
        }
    }

    _unexpected() {
        let c = this._char;
        if (c === undefined) {
            throw new json.Error('Unexpected end of JSON input.');
        } else if (c === '"') {
            c = 'string';
        } else if ((c >= '0' && c <= '9') || c === '-') {
            c = 'number';
        } else {
            if (c < ' ' || c > '\x7F') {
                const name = Object.keys(this._escape).filter((key) => this._escape[key] === c);
                c = (name.length === 1) ? `\\${name}` : `\\u${(`000${c.charCodeAt(0).toString(16)}`).slice(-4)}`;
            }
            c = `token '${c}'`;
        }
        throw new json.Error(`Unexpected ${c} ${this._location()}`);
    }

    _location() {
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
};

json.BinaryReader = class {

    static open(data) {
        return data ? new json.BinaryReader(data) : null;
    }

    constructor(data) {
        this._buffer = data instanceof Uint8Array ? data : data.peek();
    }

    read() {
        const buffer = this._buffer;
        const length = buffer.length;
        const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        const asciiDecoder = new TextDecoder('ascii');
        const utf8Decoder = new TextDecoder('utf-8');
        let position = 0;
        const skip = (offset) => {
            position += offset;
            if (position > length) {
                throw new bson.Error(`Expected ${position + length} more bytes. The file might be corrupted. Unexpected end of file.`);
            }
        };
        const header = () => {
            const start = position;
            skip(4);
            const size = view.getInt32(start, 4);
            if (size < 5 || start + size > length || buffer[start + size - 1] !== 0x00) {
                throw new bson.Error('Invalid file size.');
            }
        };
        header();
        const stack = [];
        let obj = {};
        for (;;) {
            skip(1);
            const type = buffer[position - 1];
            if (type === 0x00) {
                if (stack.length === 0) {
                    break;
                }
                obj = stack.pop();
                continue;
            }
            const start = position;
            position = buffer.indexOf(0x00, start) + 1;
            const key = asciiDecoder.decode(buffer.subarray(start, position - 1));
            let value = null;
            switch (type) {
                case 0x01: { // float64
                    const start = position;
                    skip(8);
                    value = view.getFloat64(start, true);
                    break;
                }
                case 0x02: { // string
                    skip(4);
                    const size = view.getInt32(position - 4, true);
                    const start = position;
                    skip(size);
                    value = utf8Decoder.decode(buffer.subarray(start, position - 1));
                    if (buffer[position - 1] !== 0) {
                        throw new bson.Error('String missing terminal 0.');
                    }
                    break;
                }
                case 0x03: { // object
                    header();
                    value = {};
                    break;
                }
                case 0x04: { // array
                    header();
                    value = [];
                    break;
                }
                case 0x05: { // bytes
                    const start = position;
                    skip(5);
                    const size = view.getInt32(start, true);
                    const subtype = buffer[start + 4];
                    if (subtype !== 0x00) {
                        throw new bson.Error(`Unsupported binary subtype '${subtype}'.`);
                    }
                    skip(size);
                    value = buffer.subarray(start + 5, position);
                    break;
                }
                case 0x08: { // boolean
                    skip(1);
                    value = buffer[position - 1];
                    if (value > 1) {
                        throw new bson.Error(`Invalid boolean value '${value}'.`);
                    }
                    value = value === 1 ? true : false;
                    break;
                }
                case 0x0A:
                    value = null;
                    break;
                case 0x10: {
                    const start = position;
                    skip(4);
                    value = view.getInt32(start, true);
                    break;
                }
                case 0x11: { // uint64
                    const start = position;
                    skip(8);
                    value = Number(view.getBigUint64(start, true));
                    break;
                }
                case 0x12: { // int64
                    const start = position;
                    skip(8);
                    value = Number(view.getBigInt64(start, true));
                    break;
                }
                default: {
                    throw new bson.Error(`Unsupported value type '${type}'.`);
                }
            }
            if (Array.isArray(obj))  {
                if (obj.length !== parseInt(key, 10)) {
                    throw new bson.Error(`Invalid array index '${key}'.`);
                }
                obj.push(value);
            } else {
                switch (key) {
                    case '__proto__':
                    case 'constructor':
                    case 'prototype':
                        throw new bson.Error(`Invalid key '${key}' at ${position}'.`);
                    default:
                        break;
                }
                obj[key] = value;
            }
            if (type === 0x03 || type === 0x04) {
                stack.push(obj);
                obj = value;
            }
        }
        if (position !== length) {
            throw new bson.Error(`Unexpected data at '${position}'.`);
        }
        return obj;
    }
};

json.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'JSON Error';
    }
};

bson.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'BSON Error';
    }
};

export const TextReader = json.TextReader;
export const BinaryReader = json.BinaryReader;
