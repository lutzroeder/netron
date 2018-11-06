/*jshint esversion: 6 */

var base = base || {};

base.Int64 = class {

    constructor(buffer) {
        this._buffer = buffer;
    }

    toString(radix) {
        var high = this._readInt32(4);
        var low = this._readInt32(0);
        var str = '';
        var sign = high & 0x80000000;
        if (sign) {
            high = ~high;
            low = 0x100000000 - low;
        }
        radix = radix || 10;
        while (true) {
            var mod = (high % radix) * 0x100000000 + low;
            high = Math.floor(high / radix);
            low = Math.floor(mod / radix);
            str = (mod % radix).toString(radix) + str;
            if (!high && !low) 
            {
                break;
            }
        }
        if (sign) {
            str = "-" + str;
        }
        return str;
    }

    toBuffer() {
        return this._buffer;
    }

    _readInt32(offset) {
      return (this._buffer[offset + 3] * 0x1000000) + (this._buffer[offset + 2] << 16) + (this._buffer[offset + 1] << 8) + this._buffer[offset + 0];
    }
};

base.Uint64 = class {

    constructor(buffer) {
        this._buffer = buffer;
    }

    toString(radix) {
        var high = this._readInt32(4);
        var low = this._readInt32(0);
        var str = '';
        radix = radix || 10;
        while (true) {
            var mod = (high % radix) * 0x100000000 + low;
            high = Math.floor(high / radix);
            low = Math.floor(mod / radix);
            str = (mod % radix).toString(radix) + str;
            if (!high && !low) 
            {
                break;
            }
        }
        return str;
    }

    toBuffer() {
        return this._buffer;
    }

    _readInt32(offset) {
        return (this._buffer[offset + 3] * 0x1000000) + (this._buffer[offset + 2] << 16) + (this._buffer[offset + 1] << 8) + this._buffer[offset + 0];
    }
};

if (!DataView.prototype.getFloat16) {
    DataView.prototype.getFloat16 = function(byteOffset, littleEndian) {
        var value = this.getUint16(byteOffset, littleEndian);
        var e = (value & 0x7C00) >> 10;
        var f = value & 0x03FF;
        if (e == 0) {
            f = 0.00006103515625 * (f / 1024);
        }
        else if (e == 0x1F) {
            f = f ? NaN : Infinity;
        }
        else {
            f = DataView.__float16_pow[e] * (1 + (f / 1024));  
        }
        return value & 0x8000 ? -f : f;
    };
    DataView.__float16_pow = { 
        1: 1/16384, 2: 1/8192, 3: 1/4096, 4: 1/2048, 5: 1/1024, 6: 1/512, 7: 1/256, 8: 1/128,
        9: 1/64, 10: 1/32, 11: 1/16, 12: 1/8, 13: 1/4, 14: 1/2, 15: 1, 16: 2, 
        17: 4, 18: 8, 19: 16, 20: 32, 21: 64, 22: 128, 23: 256, 24: 512,
        25: 1024, 26: 2048, 27: 4096, 28: 8192, 29: 16384, 30: 32768, 31: 65536 
        };
}

if (!DataView.prototype.setFloat16) {
    DataView.prototype.setFloat16 = function(byteOffset, value, littleEndian) {
        DataView.__float16_float[0] = value;
        value = DataView.__float16_int[0];
        var s = (value >>> 16) & 0x8000;
        var e = (value >>> 23) & 0xff;
        var f = value & 0x7fffff;
        var v = s | DataView.__float16_base[e] | (f >> DataView.__float16_shift[e]);
        this.setUint16(byteOffset, v, littleEndian);
    };
    DataView.__float16_float = new Float32Array(1);
    DataView.__float16_int = new Uint32Array(DataView.__float16_float.buffer, 0, DataView.__float16_float.length);
    DataView.__float16_base = new Uint32Array(256);
    DataView.__float16_shift = new Uint32Array(256);
    for (var i = 0; i < 256; ++i) {
        var e = i - 127;
        if (e < -27) {
            DataView.__float16_base[i] = 0x0000;
            DataView.__float16_shift[i] = 24;
        } else if (e < -14) {
            DataView.__float16_base[i] = 0x0400 >> -e - 14;
            DataView.__float16_shift[i] = -e - 1;
        } else if (e <= 15) {
            DataView.__float16_base[i] = e + 15 << 10;
            DataView.__float16_shift[i] = 13;
        } else if (e < 128) {
            DataView.__float16_base[i] = 0x7c00;
            DataView.__float16_shift[i] = 24;
        } else {
            DataView.__float16_base[i] = 0x7c00;
            DataView.__float16_shift[i] = 13;
        }
    }
}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Int64 = base.Int64;
    module.exports.Uint64 = base.Uint64;
}