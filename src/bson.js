/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental BSON JavaScript reader

var bson = {};
var long = long || { Long: require('long') };

// http://bsonspec.org/spec.html
bson.Reader = class {

    constructor(buffer) {
        this._asciiDecoder = new TextDecoder('ascii');
        this._utf8Decoder = new TextDecoder('utf-8');
        this._buffer = buffer;
        this._position = 0;
        this._view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }

    read() {
        return this.document();
    }

    document(isArray) {
        const start = this._position;
        const size = this.int32();
        if (size < 5 || start + size > this._buffer.length || this._buffer[start + size - 1] != 0x00) {
            throw new bson.Reader('Invalid BSON size.');
        }
        let element = isArray ? [] : {};
        let index = 0;
        for (;;) {
            const type = this.byte();
            if (type == 0x00) {
                break;
            }
            const key = this.cstring();
            let value = null;
            switch (type) {
                case 0x01:
                    value = this.double();
                    break
                case 0x02:
                    value = this.string();
                    break;
                case 0x03:
                    value = this.document(false);
                    break;
                case 0x04:
                    value = this.document(true);
                    break;
                case 0x05:
                    value = this.binary();
                    break;
                case 0x08:
                    value = this.boolean();
                    break;
                case 0x0A:
                    value = null;
                    break;
                case 0x10:
                    value = this.int32();
                    break;
                case 0x11:
                    value = this.uint64();
                    break;
                case 0x12:
                    value = this.int64();
                    break;
                default:
                    throw new bson.Error("Unknown value type '" + type + "'.");    
            }
            if (isArray)  {
                if (index !== parseInt(key, 10)) {
                    throw new bson.Error("Invalid array index '" + key + "'.");    
                }
                element.push(value);
                index++;
            }
            else {
                element[key] = value;
            }
        }
        return element;
    }

    cstring() {
        const end = this._buffer.indexOf(0x00, this._position);
        const value = this._asciiDecoder.decode(this._buffer.subarray(this._position, end));
        this._position = end + 1;
        return value;
    }

    string() {
        const end = this.int32() + this._position - 1;
        const value = this._utf8Decoder.decode(this._buffer.subarray(this._position, end));
        this._position = end;
        if (this.byte() != '0x00') {
            throw new bson.Error('String missing terminal 0.');
        }
        return value;
    }

    binary() {
        const size = this.int32();
        const subtype = this.byte();
        const data = this._buffer.subarray(this._position, this._position + size);
        this._position += size;
        switch (subtype) {
            case 0x00:
                return data;
            default:
                throw new bson.Error("Unknown binary subtype '" + subtype + "'.");
        }
    }
    
    boolean()  {
        const value = this.byte();
        switch (value) {
            case 0x00: return false;
            case 0x01: return true;
            default: throw new bson.Error("Invalid boolean value '" + value + "'.");
        }
    }

    byte() {
        return this._buffer[this._position++];
    }

    int32() {
        const value = this._view.getInt32(this._position, true);
        this._position += 4;
        return value;
    }

    int64() {
        const low = this._view.getUint32(this._position, true);
        const hi = this._view.getUint32(this._position + 4, true);
        this._position += 8;
        return new long.Long(low, hi, false).toNumber();
    }

    uint64() {
        const low = this._view.getUint32(this._position, true);
        const hi = this._view.getUint32(this._position + 4, true);
        this._position += 8;
        return new long.Long(low, hi, true).toNumber();
    }
}

bson.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'BSON Error';
    }
}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Reader = bson.Reader; 
}