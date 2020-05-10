/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var pickle = pickle || {};

pickle.Unpickler = class {

    constructor(buffer) {
        this._reader = new pickle.Reader(buffer, 0);
    }

    load(function_call, persistent_load) {
        let reader = this._reader;
        let marker = [];
        let stack = [];
        let memo = new Map();
        while (reader.position < reader.length) {
            const opcode = reader.byte();
            switch (opcode) {
                case pickle.OpCode.PROTO: {
                    const version = reader.byte();
                    if (version > 4) {
                        throw new pickle.Error("Unsupported protocol version '" + version + "'.");
                    }
                    break;
                }
                case pickle.OpCode.GLOBAL:
                    stack.push([ reader.line(), reader.line() ].join('.'));
                    break;
                case pickle.OpCode.STACK_GLOBAL:
                    stack.push([ stack.pop(), stack.pop() ].reverse().join('.'));
                    break;
                case pickle.OpCode.PUT: {
                    const index = parseInt(reader.line(), 10);
                    memo.set(index, stack[stack.length - 1]);
                    break;
                }
                case pickle.OpCode.OBJ: {
                    const items = stack;
                    stack = marker.pop();
                    stack.push(function_call(items.pop(), items));
                    break;
                }
                case pickle.OpCode.GET: {
                    const index = parseInt(reader.line(), 10);
                    stack.push(memo.get(index));
                    break;
                }
                case pickle.OpCode.POP:
                    stack.pop();
                    break;
                case pickle.OpCode.POP_MARK:
                    stack = marker.pop();
                    break;
                case pickle.OpCode.DUP:
                    stack.push(stack[stack.length-1]);
                    break;
                case pickle.OpCode.PERSID:
                    stack.push(persistent_load(reader.line()));
                    break;
                case pickle.OpCode.BINPERSID:
                    stack.push(persistent_load(stack.pop()));
                    break;
                case pickle.OpCode.REDUCE: {
                    const items = stack.pop();
                    const type = stack.pop();
                    stack.push(function_call(type, items));
                    break;
                }
                case pickle.OpCode.NEWOBJ: {
                    const items = stack.pop();
                    const type = stack.pop();
                    stack.push(function_call(type, items));
                    break;
                }
                case pickle.OpCode.BINGET:
                    stack.push(memo.get(reader.byte()));
                    break;
                case pickle.OpCode.LONG_BINGET:
                    stack.push(memo.get(reader.uint32()));
                    break;
                case pickle.OpCode.BINPUT:
                    memo.set(reader.byte(), stack[stack.length - 1]);
                    break;
                case pickle.OpCode.LONG_BINPUT:
                    memo.set(reader.uint32(), stack[stack.length - 1]);
                    break;
                case pickle.OpCode.BININT:
                    stack.push(reader.int32());
                    break;
                case pickle.OpCode.BININT1:
                    stack.push(reader.byte());
                    break;
                case pickle.OpCode.LONG:
                    stack.push(parseInt(reader.line(), 10));
                    break;
                case pickle.OpCode.BININT2:
                    stack.push(reader.uint16());
                    break;
                case pickle.OpCode.BINBYTES:
                    stack.push(reader.bytes(reader.int32()));
                    break;
                case pickle.OpCode.SHORT_BINBYTES:
                    stack.push(reader.bytes(reader.byte()));
                    break;
                case pickle.OpCode.FLOAT:
                    stack.push(parseFloat(reader.line()));
                    break;
                case pickle.OpCode.BINFLOAT:
                    stack.push(reader.float64());
                    break;
                case pickle.OpCode.INT: {
                    const value = reader.line();
                    if (value == '01') {
                        stack.push(true);
                    }
                    else if (value == '00') {
                        stack.push(false);
                    }
                    else {
                        stack.push(parseInt(value, 10));
                    }
                    break;
                }
                case pickle.OpCode.EMPTY_LIST:
                    stack.push([]);
                    break;
                case pickle.OpCode.EMPTY_TUPLE:
                    stack.push([]);
                    break;
                case pickle.OpCode.EMPTY_SET:
                    stack.push([]);
                    break;
                case pickle.OpCode.ADDITEMS: {
                    const items = stack;
                    stack = marker.pop();
                    let obj = stack[stack.length - 1];
                    for (let i = 0; i < items.length; i++) {
                        obj.push(items[i]);
                    }
                    break;
                }
                case pickle.OpCode.DICT: {
                    const items = stack;
                    stack = marker.pop();
                    let dict = {};
                    for (let i = 0; i < items.length; i += 2) {
                        dict[items[i]] = items[i + 1];
                    }
                    stack.push(dict);
                    break;
                }
                case pickle.OpCode.LIST: {
                    const items = stack;
                    stack = marker.pop();
                    stack.push(items);
                    break;
                }
                case pickle.OpCode.TUPLE: {
                    const items = stack;
                    stack = marker.pop();
                    stack.push(items);
                    break;
                }
                case pickle.OpCode.SETITEM: {
                    const value = stack.pop();
                    const key = stack.pop();
                    let obj = stack[stack.length - 1];
                    if (obj.__setitem__) {
                        obj.__setitem__(key, value);
                    }
                    else {
                        obj[key] = value;
                    }
                    break;
                }
                case pickle.OpCode.SETITEMS: {
                    const items = stack;
                    stack = marker.pop();
                    let obj = stack[stack.length - 1];
                    for (let i = 0; i < items.length; i += 2) {
                        if (obj.__setitem__) {
                            obj.__setitem__(items[i], items[i + 1]);
                        }
                        else {
                            obj[items[i]] = items[i + 1];
                        }
                    }
                    break;
                }
                case pickle.OpCode.EMPTY_DICT:
                    stack.push({});
                    break;
                case pickle.OpCode.APPEND: {
                    const append = stack.pop();
                    stack[stack.length-1].push(append);
                    break;
                }
                case pickle.OpCode.APPENDS: {
                    const appends = stack;
                    stack = marker.pop();
                    let list = stack[stack.length - 1];
                    list.push.apply(list, appends);
                    break;
                }
                case pickle.OpCode.STRING: {
                    const str = reader.line();
                    stack.push(str.substr(1, str.length - 2));
                    break;
                }
                case pickle.OpCode.BINSTRING:
                    stack.push(reader.string(reader.uint32()));
                    break;
                case pickle.OpCode.SHORT_BINSTRING:
                    stack.push(reader.string(reader.byte()));
                    break;
                case pickle.OpCode.UNICODE:
                    stack.push(reader.line());
                    break;
                case pickle.OpCode.BINUNICODE:
                    stack.push(reader.string(reader.uint32(), 'utf-8'));
                    break;
                case pickle.OpCode.SHORT_BINUNICODE:
                    stack.push(reader.string(reader.byte(), 'utf-8'));
                    break;
                case pickle.OpCode.BUILD: {
                    const state = stack.pop();
                    let obj = stack.pop();
                    if (obj.__setstate__) {
                        if (obj.__setstate__.__call__) {
                            obj.__setstate__.__call__([ obj, state ]);
                        }
                        else {
                            obj.__setstate__(state);
                        }
                    }
                    else {
                        for (const p in state) {
                            obj[p] = state[p];
                        }
                    }
                    if (obj.__read__) {
                        obj = obj.__read__(this);
                    }
                    stack.push(obj);
                    break;
                }
                case pickle.OpCode.MARK:
                    marker.push(stack);
                    stack = [];
                    break;
                case pickle.OpCode.NEWTRUE:
                    stack.push(true);
                    break;
                case pickle.OpCode.NEWFALSE:
                    stack.push(false);
                    break;
                case pickle.OpCode.LONG1: {
                    const data = reader.bytes(reader.byte());
                    let number = 0;
                    switch (data.length) {
                        case 0: number = 0; break;
                        case 1: number = data[0]; break;
                        case 2: number = data[1] << 8 | data[0]; break;
                        case 3: number = data[2] << 16 | data[1] << 8 | data[0]; break;
                        case 4: number = data[3] << 24 | data[2] << 16 | data[1] << 8 | data[0]; break;
                        default: number = Array.prototype.slice.call(data, 0); break;
                    }
                    stack.push(number);
                    break;
                }
                case pickle.OpCode.LONG4:
                    // TODO decode LONG4
                    stack.push(reader.bytes(reader.uint32()));
                    break;
                case pickle.OpCode.TUPLE1:
                    stack.push([ stack.pop() ]);
                    break;
                case pickle.OpCode.TUPLE2: {
                    const b = stack.pop();
                    const a = stack.pop();
                    stack.push([ a, b ]);
                    break;
                }
                case pickle.OpCode.TUPLE3: {
                    const c = stack.pop();
                    const b = stack.pop();
                    const a = stack.pop();
                    stack.push([ a, b, c ]);
                    break;
                }
                case pickle.OpCode.MEMOIZE:
                    memo.set(memo.size, stack[stack.length - 1]);
                    break;
                case pickle.OpCode.FRAME:
                    reader.bytes(8);
                    break;
                case pickle.OpCode.NONE:
                    stack.push(null);
                    break;
                case pickle.OpCode.STOP:
                    return stack.pop();
                default:
                    throw new pickle.Error("Unknown opcode '" + opcode + "'.");
            }
        }
        throw new pickle.Error('Unexpected end of file.');
    }

    read(size) {
        return this._reader.bytes(size);
    }

    unescape(token, size) {
        const length = token.length;
        const a = new Uint8Array(length);
        if (size && size == length) {
            for (let p = 0; p < size; p++) {
                a[p] = token.charCodeAt(p);
            }
            return a;
        }
        let i = 0;
        let o = 0;
        while (i < length) {
            let c = token.charCodeAt(i++);
            if (c !== 0x5C || i >= length) {
                a[o++] = c;
            }
            else {
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
                    case 0x78: { // X
                        const xsi = i - 1;
                        const xso = o;
                        for (let xi = 0; xi < 2; xi++) {
                            if (i >= length) {
                                i = xsi;
                                o = xso;
                                a[o] = 0x5c;
                                break;
                            }
                            let xd = token.charCodeAt(i++);
                            xd = xd >= 65 && xd <= 70 ? xd - 55 : xd >= 97 && xd <= 102 ? xd - 87 : xd >= 48 && xd <= 57 ? xd - 48 : -1;
                            if (xd === -1) {
                                i = xsi;
                                o = xso;
                                a[o] = 0x5c;
                                break;
                            }
                            a[o] = a[o] << 4 | xd;
                        }
                        o++;
                        break;
                    }
                    default:
                        if (c < 48 || c > 57) { // 0-9
                            a[o++] = 0x5c;
                            a[o++] = c;
                        }
                        else {
                            i--;
                            let osi = i;
                            let oso = o;
                            for (let oi = 0; oi < 3; oi++) {
                                if (i >= length) {
                                    i = osi;
                                    o = oso;
                                    a[o] = 0x5c;
                                    break;
                                }
                                let od = token.charCodeAt(i++);
                                if (od < 48 || od > 57) {
                                    i = osi;
                                    o = oso;
                                    a[o] = 0x5c;
                                    break;
                                }
                                a[o] = a[o] << 3 | od - 48;
                            }
                            o++;
                        }
                        break;
                }
            }
        }
        return a.slice(0, o);
    }
};

// https://svn.python.org/projects/python/trunk/Lib/pickletools.py
// https://github.com/python/cpython/blob/master/Lib/pickle.py
pickle.OpCode = {
    MARK: 40,              // '('
    EMPTY_TUPLE: 41,       // ')'
    STOP: 46,              // '.'
    POP: 48,               // '0'
    POP_MARK: 49,          // '1'
    DUP: 50,               // '2'
    BINBYTES: 66,          // 'B' (Protocol 3)
    SHORT_BINBYTES: 67,    // 'C' (Protocol 3)
    FLOAT: 70,             // 'F'
    BINFLOAT: 71,          // 'G'
    INT: 73,               // 'I'
    BININT: 74,            // 'J'
    BININT1: 75,           // 'K'
    LONG: 76,              // 'L'
    BININT2: 77,           // 'M'
    NONE: 78,              // 'N'
    PERSID: 80,            // 'P'
    BINPERSID: 81,         // 'Q'
    REDUCE: 82,            // 'R'
    STRING: 83,             // 'S'
    BINSTRING: 84,         // 'T'
    SHORT_BINSTRING: 85,   // 'U'
    UNICODE: 86,           // 'V'
    BINUNICODE: 88,        // 'X'
    EMPTY_LIST: 93,        // ']'
    APPEND: 97,            // 'a'
    BUILD: 98,             // 'b'
    GLOBAL: 99,            // 'c'
    DICT: 100,             // 'd'
    APPENDS: 101,          // 'e'
    GET: 103,              // 'g'
    BINGET: 104,           // 'h'
    LONG_BINGET: 106,      // 'j'
    LIST: 108,             // 'l'
    OBJ: 111,              // 'o'
    PUT: 112,              // 'p'
    BINPUT: 113,           // 'q'
    LONG_BINPUT: 114,      // 'r'
    SETITEM: 115,          // 's'
    TUPLE: 116,            // 't'
    SETITEMS: 117,         // 'u'
    EMPTY_DICT: 125,       // '}'
    PROTO: 128,
    NEWOBJ: 129,
    TUPLE1: 133,           // '\x85'
    TUPLE2: 134,           // '\x86'
    TUPLE3: 135,           // '\x87'
    NEWTRUE: 136,          // '\x88'
    NEWFALSE: 137,         // '\x89'
    LONG1: 138,            // '\x8a'
    LONG4: 139,            // '\x8b'
    SHORT_BINUNICODE: 140, // '\x8c' (Protocol 4)
    BINUNICODE8: 141,      // '\x8d' (Protocol 4)
    BINBYTES8: 142,        // '\x8e' (Protocol 4)
    EMPTY_SET: 143,        // '\x8f' (Protocol 4)
    ADDITEMS: 144,         // '\x90' (Protocol 4)
    FROZENSET: 145,        // '\x91' (Protocol 4)
    NEWOBJ_EX: 146,        // '\x92' (Protocol 4)
    STACK_GLOBAL: 147,     // '\x93' (Protocol 4)
    MEMOIZE: 148,          // '\x94' (Protocol 4)
    FRAME: 149             // '\x95' (Protocol 4)
};

pickle.Reader = class {

    constructor(buffer) {
        if (buffer) {
            this._buffer = buffer;
            this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
            this._position = 0;
        }
        pickle.Reader._utf8Decoder = pickle.Reader._utf8Decoder || new TextDecoder('utf-8');
        pickle.Reader._asciiDecoder = pickle.Reader._asciiDecoder || new TextDecoder('ascii');
    }

    get length() {
        return this._buffer.byteLength;
    }

    get position() {
        return this._position;
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._dataView.getUint8(position);
    }

    bytes(length) {
        const position = this._position;
        this.skip(length);
        return this._buffer.subarray(position, this._position);
    }

    uint16() {
        const position = this.position;
        this.skip(2);
        return this._dataView.getUint16(position, true);
    }

    int32() {
        const position = this.position;
        this.skip(4);
        return this._dataView.getInt32(position, true);
    }

    uint32() {
        const position = this.position;
        this.skip(4);
        return this._dataView.getUint32(position, true);
    }

    float32() {
        const position = this.position;
        this.skip(4);
        return this._dataView.getFloat32(position, true);
    }

    float64() {
        const position = this.position;
        this.skip(8);
        return this._dataView.getFloat64(position, true);
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new pickle.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    string(size, encoding) {
        const data = this.bytes(size);
        return (encoding == 'utf-8') ?
            pickle.Reader._utf8Decoder.decode(data) :
            pickle.Reader._asciiDecoder.decode(data);
    }

    line() {
        const index = this._buffer.indexOf(0x0A, this._position);
        if (index == -1) {
            throw new pickle.Error("Could not find end of line.");
        }
        const size = index - this._position;
        const text = this.string(size, 'ascii');
        this.skip(1);
        return text;
    }
};


pickle.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Unpickle Error';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Unpickler = pickle.Unpickler;
}
