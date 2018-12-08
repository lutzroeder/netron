/*jshint esversion: 6 */

var pickle = pickle || {};

pickle.Unpickler = class {

    constructor(buffer) {
        this._reader = new pickle.Reader(buffer, 0);
    }

    load(function_call, persistent_load) {
        var reader = this._reader;
        var stack = [];
        var marker = [];
        var table = {};    
        while (reader.position < reader.length) {
            var opcode = reader.byte();
            // console.log(reader.position.toString() + ': ' + opcode + ' ' + String.fromCharCode(opcode));
            switch (opcode) {
                case pickle.OpCode.PROTO:
                    var version = reader.byte();
                    if (version > 4) {
                        throw new pickle.Error("Unsupported protocol version '" + version + "'.");
                    }
                    break;
                case pickle.OpCode.GLOBAL:
                    stack.push([ reader.line(), reader.line() ].join('.'));
                    break;
                case pickle.OpCode.PUT:
                    table[reader.line()] = stack[stack.length - 1];
                    break;
                case pickle.OpCode.OBJ:
                    var obj = new (stack.pop())();
                    var index = marker.pop();
                    for (var position = index; position < stack.length; pos += 2) {
                        obj[stack[position]] = stack[position + 1];
                    }
                    stack = stack.slice(0, index);
                    stack.push(obj);
                    break;
                case pickle.OpCode.GET:
                    stack.push(table[reader.line()]);
                    break;
                case pickle.OpCode.POP:
                    stack.pop();
                    break;
                case pickle.OpCode.POP_MARK:
                    stack = stack.slice(0, marker.pop());
                    break;
                case pickle.OpCode.DUP:
                    var value = stack[stack.length-1];
                    stack.push(value);
                    break;
                case pickle.OpCode.PERSID:
                    stack.push(persistent_load(reader.line()));
                    break;
                case pickle.OpCode.BINPERSID:
                    stack.push(persistent_load(stack.pop()));
                    break;
                case pickle.OpCode.REDUCE:
                    var args = stack.pop();
                    var type = stack.pop();
                    stack.push(function_call(type, args));
                    break;
                case pickle.OpCode.NEWOBJ:
                    var args = stack.pop();
                    var type = stack.pop();
                    stack.push(function_call(type, args));
                    break;
                case pickle.OpCode.BINGET:
                    stack.push(table[reader.byte()]);
                    break;
                case pickle.OpCode.LONG_BINGET:
                    stack.push(table[reader.uint32()]);
                    break;
                case pickle.OpCode.BINPUT:
                    table[reader.byte()] = stack[stack.length - 1];
                    break;
                case pickle.OpCode.LONG_BINPUT:
                    table[reader.uint32()] = stack[stack.length - 1];
                    break;
                case pickle.OpCode.BININT:
                    stack.push(reader.int32());
                    break;
                case pickle.OpCode.BININT1:
                    stack.push(reader.byte());
                    break;
                case pickle.OpCode.LONG:
                    stack.push(parseInt(reader.line()));
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
                case pickle.OpCode.INT:
                    var value = reader.line();
                    if (value == '01') {
                        stack.push(true);
                    }
                    else if (value == '00') {
                        stack.push(false);
                    }
                    else {
                        stack.push(parseInt(value));
                    }
                    break;
                case pickle.OpCode.EMPTY_LIST:
                    stack.push([]);
                    break;
                case pickle.OpCode.EMPTY_TUPLE:
                    stack.push([]);
                    break;
                case pickle.OpCode.DICT:
                    var index = marker.pop();
                    var obj = {};
                    for (var position = index; position < stack.length; position += 2) {
                        obj[stack[position]] = stack[position + 1];
                    }
                    stack = stack.slice(0, index);
                    stack.push(obj);
                    break;
                case pickle.OpCode.LIST:
                    stack.push(stack.splice(marker.pop()));
                    break;
                case pickle.OpCode.TUPLE:
                    stack.push(stack.splice(marker.pop()));
                    break;
                case pickle.OpCode.SETITEM:
                    var value = stack.pop();
                    var key = stack.pop();
                    var obj = stack[stack.length - 1];
                    if (obj.__setitem__) {
                        obj.__setitem__(key, value);
                    }
                    else {
                        obj[key] = value;
                    }
                    break;
                case pickle.OpCode.SETITEMS:
                    var index = marker.pop();
                    var obj = stack[index - 1];
                    if (obj.__setitem__) {
                        for (var position = index; position < stack.length; position += 2) {
                            obj.__setitem__(stack[position], stack[position + 1]);
                        }
                    }
                    else {
                        for (var position = index; position < stack.length; position += 2) {
                            obj[stack[position]] = stack[position + 1];
                        }
                    }
                    stack = stack.slice(0, index);
                    break;
                case pickle.OpCode.EMPTY_DICT:
                    stack.push({});
                    break;
                case pickle.OpCode.APPEND:
                    var append = stack.pop();
                    stack[stack.length-1].push(append);
                    break;
                case pickle.OpCode.APPENDS:
                    var appends = stack.splice(marker.pop());
                    var list = stack[stack.length - 1];
                    list.push.apply(list, appends);
                    break;
                case pickle.OpCode.STRING:
                    var value = reader.line();
                    stack.push(value.substr(1, value.length - 2));
                    break;
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
                case pickle.OpCode.BUILD:
                    var dict = stack.pop();
                    var obj = stack.pop();
                    if (obj.__setstate__) {
                        obj.__setstate__(dict);
                    }
                    else {
                        for (var p in dict) {
                            obj[p] = dict[p];
                        }
                    }
                    if (obj.__read__) {
                        obj = obj.__read__(this);
                    }
                    stack.push(obj);
                    break;
                case pickle.OpCode.MARK:
                    marker.push(stack.length);
                    break;
                case pickle.OpCode.NEWTRUE:
                    stack.push(true);
                    break;
                case pickle.OpCode.NEWFALSE:
                    stack.push(false);
                    break;
                case pickle.OpCode.LONG1:
                    var data = reader.bytes(reader.byte());
                    var value = 0;
                    switch (data.length) {
                        case 0: value = 0; break;
                        case 1: value = data[0]; break;
                        case 2: value = data[1] << 8 | data[0]; break;
                        case 3: value = data[2] << 16 | data[1] << 8 | data[0]; break;
                        case 4: value = data[3] << 24 | data[2] << 16 | data[0]; break;
                        default: value = Array.prototype.slice.call(data, 0); break; 
                    }
                    stack.push(value);
                    break;
                case pickle.OpCode.LONG4:
                    // TODO decode LONG4
                    var data = reader.bytes(reader.uint32());
                    stack.push(data);
                    break;
                case pickle.OpCode.TUPLE1:
                    var a = stack.pop();
                    stack.push([a]);
                    break;
                case pickle.OpCode.TUPLE2:
                    var b = stack.pop();
                    var a = stack.pop();
                    stack.push([a, b]);
                    break;
                case pickle.OpCode.TUPLE3:
                    var c = stack.pop();
                    var b = stack.pop();
                    var a = stack.pop();
                    stack.push([a, b, c]);
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
        var length = token.length;
        var a = new Uint8Array(length);
        if (size && size == length) {
            for (var p = 0; p < size; p++) {
                a[p] = token.charCodeAt(p);
            }
            return a;
        }
        var i = 0;
        var o = 0;
        while (i < length) {
            var c = token.charCodeAt(i++);
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
                    case 0x78: // X
                        var xsi = i - 1;
                        var xso = o;
                        for (var xi = 0; xi < 2; xi++) {
                            if (i >= length) {
                                i = xsi;
                                o = xso;
                                a[o] = 0x5c;
                                break;
                            }
                            var xd = token.charCodeAt(i++);
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
                    default:
                        if (c < 48 || c > 57) { // 0-9
                            a[o++] = 0x5c;
                            a[o++] = c;
                        }
                        else {
                            i--;
                            var osi = i;
                            var oso = o;
                            for (var oi = 0; oi < 3; oi++) {
                                if (i >= length) {
                                    i = osi;
                                    o = oso;
                                    a[o] = 0x5c;
                                    break;
                                }
                                var od = token.charCodeAt(i++);
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

           if (token.length == 65536 && i != o) {
               debugger;
           }
        }
        return a.slice(0, o);
    }
};

// https://svn.python.org/projects/python/trunk/Lib/pickletools.py
// https://github.com/python/cpython/blob/master/Lib/pickle.py
pickle.OpCode = {
    MARK: 40,            // '('
    EMPTY_TUPLE: 41,     // ')'
    STOP: 46,            // '.'
    POP: 48,             // '0'
    POP_MARK: 49,        // '1'
    DUP: 50,             // '2'
    BINBYTES: 66,        // 'B' (Protocol 3)
    SHORT_BINBYTES: 67,  // 'C' (Protocol 3)
    FLOAT: 70,           // 'F'
    BINFLOAT: 71,        // 'G'
    INT: 73,             // 'I'
    BININT: 74,          // 'J'
    BININT1: 75,         // 'K'
    LONG: 76,            // 'L'
    BININT2: 77,         // 'M'
    NONE: 78,            // 'N'
    PERSID: 80,          // 'P'
    BINPERSID: 81,       // 'Q'
    REDUCE: 82,          // 'R'
    STRING: 83,          // 'S'
    BINSTRING: 84,       // 'T'
    SHORT_BINSTRING: 85, // 'U'
    UNICODE: 86,         // 'V'
    BINUNICODE: 88,      // 'X'
    EMPTY_LIST: 93,      // ']'
    APPEND: 97,          // 'a'
    BUILD: 98,           // 'b'
    GLOBAL: 99,          // 'c'
    DICT: 100,           // 'd'
    APPENDS: 101,        // 'e'
    GET: 103,            // 'g'
    BINGET: 104,         // 'h'
    LONG_BINGET: 106,    // 'j'
    LIST: 108,           // 'l'
    OBJ: 111,            // 'o'
    PUT: 112,            // 'p'
    BINPUT: 113,         // 'q'
    LONG_BINPUT: 114,    // 'r'
    SETITEM: 115,        // 's'
    TUPLE: 116,          // 't'
    SETITEMS: 117,       // 'u'
    EMPTY_DICT: 125,     // '}'
    PROTO: 128,
    NEWOBJ: 129,
    TUPLE1: 133,         // '\x85'
    TUPLE2: 134,         // '\x86'
    TUPLE3: 135,         // '\x87'
    NEWTRUE: 136,        // '\x88'
    NEWFALSE: 137,       // '\x89'
    LONG1: 138,          // '\x8a'
    LONG4: 139           // '\x8b'
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
        var value = this._dataView.getUint8(this._position);
        this._position++;
        return value;
    }

    bytes(length) {
        var data = this._buffer.subarray(this._position, this._position + length);
        this._position += length;
        return data;
    }

    uint16() {
        var value = this._dataView.getUint16(this._position, true);
        this._position += 2;
        return value;
    }

    int32() {
        var value = this._dataView.getInt32(this._position, true);
        this._position += 4;
        return value;
    }

    uint32() {
        var value = this._dataView.getUint32(this._position, true);
        this._position += 4;
        return value;
    }

    float32() {
        var value = this._dataView.getFloat32(this._position, true);
        this._position += 4;
        return value;
    }

    float64() {
        var value = this._dataView.getFloat64(this._position, false);
        this._position += 8;
        return value;
    }

    seek(offset) {
        this._position += offset;
    }

    string(size, encoding) {
        var data = this.bytes(size);
        var text = (encoding == 'utf-8') ?
            pickle.Reader._utf8Decoder.decode(data) :
            pickle.Reader._asciiDecoder.decode(data);
        return text;
    }

    line() {
        var index = this._buffer.indexOf(0x0A, this._position);
        if (index == -1) {
            throw new pickle.Error("Could not find end of line.");
        }
        var size = index - this._position;
        var text = this.string(size, 'ascii');
        this.seek(1);
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
