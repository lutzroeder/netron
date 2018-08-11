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
        while (reader.offset < reader.length) {
            var opcode = reader.readByte();
            // console.log(reader.offset.toString() + ': ' + opcode + ' ' + String.fromCharCode(opcode));
            switch (opcode) {
                case pickle.OpCode.PROTO:
                    var version = reader.readByte();
                    if (version != 2) {
                        throw new pickle.Error("Unsupported protocol version '" + version + "'.");
                    }
                    break;
                case pickle.OpCode.GLOBAL:
                    stack.push([ reader.readLine(), reader.readLine() ].join('.'));
                    break;
                case pickle.OpCode.PUT:
                    table[reader.readLine()] = stack[stack.length - 1];
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
                    stack.push(table[reader.readLine()]);
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
                    throw new pickle.Error("Unknown opcode 'PERSID'.");
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
                    stack.push(table[reader.readByte()]);
                    break;
                case pickle.OpCode.LONG_BINGET:
                    stack.push(table[reader.readUInt32()]);
                    break;
                case pickle.OpCode.BINPUT:
                    table[reader.readByte()] = stack[stack.length - 1];
                    break;
                case pickle.OpCode.LONG_BINPUT:
                    table[reader.readUInt32()] = stack[stack.length - 1];
                    break;
                case pickle.OpCode.BININT:
                    stack.push(reader.readInt32());
                    break;
                case pickle.OpCode.BININT1:
                    stack.push(reader.readByte());
                    break;
                case pickle.OpCode.LONG:
                    stack.push(parseInt(reader.readLine()));
                    break;
                case pickle.OpCode.BININT2:
                    stack.push(reader.readUInt16());
                    break;
                case pickle.OpCode.FLOAT:
                    stack.push(parseFloat(reader.readLine()));
                    break;
                case pickle.OpCode.BINFLOAT:
                    stack.push(reader.readFloat64());
                    break;
                case pickle.OpCode.INT:
                    var value = reader.readLine();
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
                    obj[key] = value;
                    break;
                case pickle.OpCode.SETITEMS:
                    var index = marker.pop();
                    var obj = stack[index - 1];
                    for (var position = index; position < stack.length; position += 2) {
                        obj[stack[position]] = stack[position + 1];
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
                    var value = reader.readLine();
                    stack.push(value.substr(1, value.length - 2));
                    break;
                case pickle.OpCode.BINSTRING:
                    stack.push(reader.readString(reader.readUInt32()));
                    break;
                case pickle.OpCode.SHORT_BINSTRING:
                    stack.push(reader.readString(reader.readByte()));
                    break;
                case pickle.OpCode.UNICODE:
                    stack.push(reader.readLine());
                    break;
                case pickle.OpCode.BINUNICODE:
                    stack.push(reader.readString(reader.readUInt32(), 'utf-8'));
                    break;
                case pickle.OpCode.BUILD:
                    var dict = stack.pop();
                    var obj = stack.pop();
                    for (var p in dict) {
                        obj[p] = dict[p];
                    }
                    stack.push(obj);
                    if (obj.unpickle) {
                        obj.unpickle(reader);
                    }
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
                    var data = reader.readBytes(reader.readByte());
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
                    var data = reader.readBytes(reader.readUInt32());
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
        return this._reader.readBytes(size);
    }
};

// https://svn.python.org/projects/python/trunk/Lib/pickletools.py
pickle.OpCode = {
    MARK: 40,            // '('
    EMPTY_TUPLE: 41,     // ')'
    STOP: 46,            // '.'
    POP: 48,             // '0'
    POP_MARK: 49,        // '1'
    DUP: 50,             // '2'
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
            this._offset = 0;
        }
        pickle.Reader._utf8Decoder = pickle.Reader._utf8Decoder || new TextDecoder('utf-8');
        pickle.Reader._asciiDecoder = pickle.Reader._asciiDecoder || new TextDecoder('ascii');
    }

    get length() {
        return this._buffer.byteLength;
    }

    get offset() {
        return this._offset;
    }

    readByte() {
        var value = this._dataView.getUint8(this._offset);
        this._offset++;
        return value;
    }

    readBytes(length) {
        var data = this._buffer.subarray(this._offset, this._offset + length);
        this._offset += length;
        return data;
    }

    readUInt16() {
        var value = this._dataView.getUint16(this._offset, true);
        this._offset += 2;
        return value;
    }

    readInt32() {
        var value = this._dataView.getInt32(this._offset, true);
        this._offset += 4;
        return value;
    }

    readUInt32() {
        var value = this._dataView.getUint32(this._offset, true);
        this._offset += 4;
        return value;
    }

    readFloat32() {
        var value = this._dataView.getFloat32(this._offset, true);
        this._offset += 4;
        return value;
    }

    readFloat64() {
        var value = this._dataView.getFloat64(this._offset, false);
        this._offset += 8;
        return value;
    }

    skipBytes(length) {
        this._offset += length;
    }

    readString(size, encoding) {
        var data = this.readBytes(size);
        var text = '';
        if (encoding == 'utf-8') {
            text = pickle.Reader._utf8Decoder.decode(data);    
        }
        else {
            text = pickle.Reader._asciiDecoder.decode(data);
        }
        return text.replace(/\0/g, '');
    }

    readLine() {
        var index = this._buffer.indexOf(0x0A, this._offset);
        if (index == -1) {
            throw new pickle.Error("Could not find end of line.");
        }
        var size = index - this._offset;
        var text = this.readString(size, 'ascii');
        this.skipBytes(1);
        return text;
    }
};


pickle.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Unpickle Error';
    }
};
