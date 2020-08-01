/* jshint esversion: 6 */

var base = base || {};

base.Int64 = class Int64 {

    constructor(low, high) {
        this.low = low;
        this.high = high;
    }

    toNumber() {
        return (this.high * 4294967296) + (this.low >>> 0);
    }

    toString(radix) {
        radix = radix || 10;
        if (radix < 2 || 36 < radix) {
            throw RangeError('radix');
        }
        if (this.high === 0 && this.low === 0) {
            return '0';
        }
        if (this.high < 0) {
            throw new Error('Not implemented.');
        }
        throw new Error('Not implemented.');
    }
};

base.Uint64 = class Uint64 {

    constructor(low, high) {
        this.low = low;
        this.high = high;
    }

    toNumber() {
        return ((this.high >>> 0) * 4294967296) + (this.low >>> 0);
    }

    toString(radix) {
        if (radix < 2 || 36 < radix) {
            throw RangeError('radix');
        }
        if (this.high === 0 && this.low === 0) {
            return '0';
        }
        throw new Error('Not implemented.');
    }
};

if (!DataView.prototype.getFloat16) {
    DataView.prototype.getFloat16 = function(byteOffset, littleEndian) {
        const value = this.getUint16(byteOffset, littleEndian);
        const e = (value & 0x7C00) >> 10;
        let f = value & 0x03FF;
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
        const s = (value >>> 16) & 0x8000;
        const e = (value >>> 23) & 0xff;
        const f = value & 0x7fffff;
        const v = s | DataView.__float16_base[e] | (f >> DataView.__float16_shift[e]);
        this.setUint16(byteOffset, v, littleEndian);
    };
    DataView.__float16_float = new Float32Array(1);
    DataView.__float16_int = new Uint32Array(DataView.__float16_float.buffer, 0, DataView.__float16_float.length);
    DataView.__float16_base = new Uint32Array(256);
    DataView.__float16_shift = new Uint32Array(256);
    for (let i = 0; i < 256; ++i) {
        const e = i - 127;
        if (e < -27) {
            DataView.__float16_base[i] = 0x0000;
            DataView.__float16_shift[i] = 24;
        }
        else if (e < -14) {
            DataView.__float16_base[i] = 0x0400 >> -e - 14;
            DataView.__float16_shift[i] = -e - 1;
        }
        else if (e <= 15) {
            DataView.__float16_base[i] = e + 15 << 10;
            DataView.__float16_shift[i] = 13;
        }
        else if (e < 128) {
            DataView.__float16_base[i] = 0x7c00;
            DataView.__float16_shift[i] = 24;
        }
        else {
            DataView.__float16_base[i] = 0x7c00;
            DataView.__float16_shift[i] = 13;
        }
    }
}

DataView.prototype.getInt64 = DataView.prototype.getInt64 || function(byteOffset, littleEndian) {
    return littleEndian ?
        new base.Int64(this.getUint32(byteOffset, true), this.getUint32(byteOffset + 4, true)) :
        new base.Int64(this.getUint32(byteOffset + 4, true), this.getUint32(byteOffset, true));
};

DataView.prototype.setInt64 = DataView.prototype.setInt64 || function(byteOffset, value, littleEndian) {
    if (littleEndian) {
        this.setUInt32(byteOffset, value.high, true);
        this.setUInt32(byteOffset + 4, value.low, true);
    }
    else {
        this.setUInt32(byteOffset + 4, value.high, false);
        this.setUInt32(byteOffset, value.low, false);
    }
};

DataView.prototype.getUint64 = DataView.prototype.getUint64 || function(byteOffset, littleEndian) {
    return littleEndian ?
        new base.Uint64(this.getUint32(byteOffset, true), this.getUint32(byteOffset + 4, true)) :
        new base.Uint64(this.getUint32(byteOffset + 4, true), this.getUint32(byteOffset, true));
};

DataView.prototype.setUint64 = DataView.prototype.setUint64 || function(byteOffset, value, littleEndian) {
    if (littleEndian) {
        this.setUInt32(byteOffset, value.high, true);
        this.setUInt32(byteOffset + 4, value.low, true);
    }
    else {
        this.setUInt32(byteOffset + 4, value.high, false);
        this.setUInt32(byteOffset, value.low, false);
    }
};

DataView.prototype.getBits = DataView.prototype.getBits || function(offset, bits /*, signed */) {
    offset = offset * bits;
    const available = (this.byteLength << 3) - offset;
    if (bits > available) {
        throw new RangeError();
    }
    let value = 0;
    let index = 0;
    while (index < bits) {
        const remainder = offset & 7;
        const size = Math.min(bits - index, 8 - remainder);
        value <<= size;
        value |= (this.getUint8(offset >> 3) >> (8 - size - remainder)) & ~(0xff << size);
        offset += size;
        index += size;
    }
    return value;
};

if (typeof window !== 'undefined' && typeof window.Long != 'undefined') {
    window.long = { Long: window.Long };
    window.Int64 = base.Int64;
    window.Uint64 = base.Uint64;
}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Int64 = base.Int64;
    module.exports.Uint64 = base.Uint64;
}
