
var base = base || {};

base.Int64 = class Int64 {

    constructor(low, high) {
        this.low = low | 0;
        this.high = high | 0;
    }

    static create(value) {
        if (isNaN(value)) {
            return base.Int64.zero;
        }
        if (value <= -9223372036854776000) {
            return base.Int64.min;
        }
        if (value + 1 >= 9223372036854776000) {
            return base.Int64.max;
        }
        if (value < 0) {
            return base.Int64.create(-value).negate();
        }
        return new base.Int64((value % 4294967296) | 0, (value / 4294967296));
    }

    get isZero() {
        return this.low === 0 && this.high === 0;
    }

    get isNegative() {
        return this.high < 0;
    }

    negate() {
        if (this.equals(base.Int64.min)) {
            return base.Int64.min;
        }
        return this.not().add(base.Int64.one);
    }

    not() {
        return new base.Int64(~this.low, ~this.high);
    }

    equals(other) {
        if (!(other instanceof base.Int64) && (this.high >>> 31) === 1 && (other.high >>> 31) === 1) {
            return false;
        }
        return this.high === other.high && this.low === other.low;
    }

    compare(other) {
        if (this.equals(other)) {
            return 0;
        }
        const thisNeg = this.isNegative;
        const otherNeg = other.isNegative;
        if (thisNeg && !otherNeg) {
            return -1;
        }
        if (!thisNeg && otherNeg) {
            return 1;
        }
        return this.subtract(other).isNegative ? -1 : 1;
    }

    add(other) {
        return base.Utility.add(this, other, false);
    }

    subtract(other) {
        return base.Utility.subtract(this, other, false);
    }

    multiply(other) {
        return base.Utility.multiply(this, other, false);
    }

    divide(other) {
        return base.Utility.divide(this, other, false);
    }

    toInteger() {
        return this.low;
    }

    toNumber() {
        if (this.high === 0) {
            return this.low >>> 0;
        }
        if (this.high === -1) {
            return this.low;
        }
        return (this.high * 4294967296) + (this.low >>> 0);
    }

    toString(radix) {
        const r = radix || 10;
        if (r < 2 || r > 16) {
            throw new RangeError('radix');
        }
        if (this.isZero) {
            return '0';
        }
        if (this.high < 0) {
            if (this.equals(base.Int64.min)) {
                const radix = new base.Int64(r, 0);
                const div = this.divide(radix);
                const remainder = div.multiply(radix).subtract(this);
                return div.toString(radix) + (remainder.low >>> 0).toString(radix);
            }
            return '-' + this.negate().toString(r);
        }
        if (this.high === 0) {
            return this.low.toString(radix);
        }
        return base.Utility.text(this, false, r);
    }
};

base.Int64.min = new base.Int64(0, -2147483648);
base.Int64.zero = new base.Int64(0, 0);
base.Int64.one = new base.Int64(1, 0);
base.Int64.negativeOne = new base.Int64(-1, 0);
base.Int64.power24 = new base.Int64(1 << 24, 0);
base.Int64.max = new base.Int64(0, 2147483647);

base.Uint64 = class Uint64 {

    constructor(low, high) {
        this.low = low | 0;
        this.high = high | 0;
    }

    static create(value) {
        if (isNaN(value)) {
            return base.Uint64.zero;
        }
        if (value < 0) {
            return base.Uint64.zero;
        }
        if (value >= 18446744073709552000) {
            return base.Uint64.max;
        }
        if (value < 0) {
            return base.Uint64.create(-value).negate();
        }
        return new base.Uint64((value % 4294967296) | 0, (value / 4294967296));
    }

    get isZero() {
        return this.low === 0 && this.high === 0;
    }

    get isNegative() {
        return false;
    }

    negate() {
        return this.not().add(base.Int64.one);
    }

    not() {
        return new base.Uint64(~this.low, ~this.high);
    }

    equals(other) {
        if (!(other instanceof base.Uint64) && (this.high >>> 31) === 1 && (other.high >>> 31) === 1) {
            return false;
        }
        return this.high === other.high && this.low === other.low;
    }

    compare(other) {
        if (this.equals(other)) {
            return 0;
        }
        const thisNeg = this.isNegative;
        const otherNeg = other.isNegative;
        if (thisNeg && !otherNeg) {
            return -1;
        }
        if (!thisNeg && otherNeg) {
            return 1;
        }
        return (other.high >>> 0) > (this.high >>> 0) || (other.high === this.high && (other.low >>> 0) > (this.low >>> 0)) ? -1 : 1;
    }

    add(other) {
        return base.Utility.add(this, other, true);
    }

    subtract(other) {
        return base.Utility.subtract(this, other, true);
    }

    multiply(other) {
        return base.Utility.multiply(this, other, true);
    }

    divide(other) {
        return base.Utility.divide(this, other, true);
    }

    toInteger() {
        return this.low >>> 0;
    }

    toNumber() {
        if (this.high === 0) {
            return this.low >>> 0;
        }
        return ((this.high >>> 0) * 4294967296) + (this.low >>> 0);
    }

    toString(radix) {
        const r = radix || 10;
        if (r < 2 || 36 < r) {
            throw new RangeError('radix');
        }
        if (this.isZero) {
            return '0';
        }
        if (this.high === 0) {
            return this.low.toString(radix);
        }
        return base.Utility.text(this, true, r);
    }
};

base.Utility = class {

    static add(a, b, unsigned) {
        const a48 = a.high >>> 16;
        const a32 = a.high & 0xFFFF;
        const a16 = a.low >>> 16;
        const a00 = a.low & 0xFFFF;
        const b48 = b.high >>> 16;
        const b32 = b.high & 0xFFFF;
        const b16 = b.low >>> 16;
        const b00 = b.low & 0xFFFF;
        let c48 = 0;
        let c32 = 0;
        let c16 = 0;
        let c00 = 0;
        c00 += a00 + b00;
        c16 += c00 >>> 16;
        c00 &= 0xFFFF;
        c16 += a16 + b16;
        c32 += c16 >>> 16;
        c16 &= 0xFFFF;
        c32 += a32 + b32;
        c48 += c32 >>> 16;
        c32 &= 0xFFFF;
        c48 += a48 + b48;
        c48 &= 0xFFFF;
        return base.Utility._create((c16 << 16) | c00, (c48 << 16) | c32, unsigned);
    }

    static subtract(a, b, unsigned) {
        return base.Utility.add(a, b.negate(), unsigned);
    }

    static multiply(a, b, unsigned) {
        if (a.isZero) {
            return base.Int64.zero;
        }
        if (b.isZero) {
            return base.Int64.zero;
        }
        if (a.equals(base.Int64.min)) {
            return b.isOdd() ? base.Int64.min : base.Int64.zero;
        }
        if (b.equals(base.Int64.min)) {
            return a.isOdd() ? base.Int64.min : base.Int64.zero;
        }
        if (a.isNegative) {
            if (b.isNegative) {
                return a.negate().multiply(b.negate());
            }
            return a.negate().multiply(b).negate();
        } else if (b.isNegative) {
            return a.multiply(b.negate()).negate();
        }
        if (a.compare(base.Int64.power24) < 0 && b.compare(base.Int64.power24) < 0) {
            return unsigned ? base.Uint64.create(a.toNumber() * b.toNumber()) : base.Int64.create(a.toNumber() * b.toNumber());
        }
        const a48 = a.high >>> 16;
        const a32 = a.high & 0xFFFF;
        const a16 = a.low >>> 16;
        const a00 = a.low & 0xFFFF;
        const b48 = b.high >>> 16;
        const b32 = b.high & 0xFFFF;
        const b16 = b.low >>> 16;
        const b00 = b.low & 0xFFFF;
        let c48 = 0;
        let c32 = 0;
        let c16 = 0;
        let c00 = 0;
        c00 += a00 * b00;
        c16 += c00 >>> 16;
        c00 &= 0xFFFF;
        c16 += a16 * b00;
        c32 += c16 >>> 16;
        c16 &= 0xFFFF;
        c16 += a00 * b16;
        c32 += c16 >>> 16;
        c16 &= 0xFFFF;
        c32 += a32 * b00;
        c48 += c32 >>> 16;
        c32 &= 0xFFFF;
        c32 += a16 * b16;
        c48 += c32 >>> 16;
        c32 &= 0xFFFF;
        c32 += a00 * b32;
        c48 += c32 >>> 16;
        c32 &= 0xFFFF;
        c48 += a48 * b00 + a32 * b16 + a16 * b32 + a00 * b48;
        c48 &= 0xFFFF;
        return base.Utility._create((c16 << 16) | c00, (c48 << 16) | c32, unsigned);
    }

    static divide(a, b, unsigned) {
        if (b.isZero) {
            throw new Error('Division by zero.');
        }
        if (a.isZero) {
            return unsigned ? base.Uint64.zero : base.Int64.zero;
        }
        let approx;
        let remainder;
        let result;
        if (!unsigned) {
            if (a.equals(base.Int64.min)) {
                if (b.equals(base.Int64.one) || b.equals(base.Int64.negativeOne)) {
                    return base.Int64.min;
                } else if (b.equals(base.Int64.min)) {
                    return base.Int64.one;
                }
                const half = base.Utility._shiftRight(a, unsigned, 1);
                const halfDivide = half.divide(b);
                approx = base.Utility._shiftLeft(halfDivide, halfDivide instanceof base.Uint64, 1);
                if (approx.equals(base.Int64.zero)) {
                    return b.isNegative ? base.Int64.one : base.Int64.negativeOne;
                }
                remainder = a.subtract(b.multiply(approx));
                result = approx.add(remainder.divide(b));
                return result;
            } else if (b.equals(base.Int64.min)) {
                return base.Int64.zero;
            }
            if (a.isNegative) {
                if (b.isNegative) {
                    return this.negate().divide(b.negate());
                }
                return a.negate().divide(b).negate();
            } else if (b.isNegative) {
                return a.divide(b.negate()).negate();
            }
            result = base.Int64.zero;
        } else {
            if (!(b instanceof base.Uint64)) {
                b = new base.Uint64(b.low, b.high);
            }
            if (b.compare(a) > 0) {
                return base.Int64.zero;
            }
            if (b.compare(base.Utility._shiftRight(a, unsigned, 1)) > 0) {
                return base.Uint64.one;
            }
            result = base.Uint64.zero;
        }
        remainder = a;
        while (remainder.compare(b) >= 0) {
            let approx = Math.max(1, Math.floor(remainder.toNumber() / b.toNumber()));
            const log2 = Math.ceil(Math.log(approx) / Math.LN2);
            const delta = (log2 <= 48) ? 1 : Math.pow(2, log2 - 48);
            let approxResult = base.Int64.create(approx);
            let approxRemainder = approxResult.multiply(b);
            while (approxRemainder.isNegative || approxRemainder.compare(remainder) > 0) {
                approx -= delta;
                approxResult = unsigned ? base.Uint64.create(approx) : base.Int64.create(approx);
                approxRemainder = approxResult.multiply(b);
            }
            if (approxResult.isZero) {
                approxResult = base.Int64.one;
            }
            result = result.add(approxResult);
            remainder = remainder.subtract(approxRemainder);
        }
        return result;
    }

    static text(value, unsigned, radix) {
        const power = unsigned ? base.Uint64.create(Math.pow(radix, 6)) : base.Int64.create(Math.pow(radix, 6));
        let remainder = value;
        let result = '';
        for (;;) {
            const remainderDiv = remainder.divide(power);
            const intval = remainder.subtract(remainderDiv.multiply(power)).toInteger() >>> 0;
            let digits = intval.toString(radix);
            remainder = remainderDiv;
            if (remainder.low === 0 && remainder.high === 0) {
                return digits + result;
            }
            while (digits.length < 6) {
                digits = '0' + digits;
            }
            result = '' + digits + result;
        }
    }

    static _shiftLeft(value, unsigned, shift) {
        return base.Utility._create(value.low << shift, (value.high << shift) | (value.low >>> (32 - shift)), unsigned);
    }

    static _shiftRight(value, unsigned, shift) {
        return base.Utility._create((value.low >>> shift) | (value.high << (32 - shift)), value.high >> shift, unsigned);
    }

    static _create(low, high, unsigned) {
        return unsigned ? new base.Uint64(low, high) : new base.Int64(low, high);
    }
};

base.Uint64.zero = new base.Uint64(0, 0);
base.Uint64.one = new base.Uint64(1, 0);
base.Uint64.max = new base.Uint64(-1, -1);

base.Complex64 = class Complex {

    constructor(real, imaginary) {
        this.real = real;
        this.imaginary = imaginary;
    }

    static create(real, imaginary) {
        return new base.Complex64(real, imaginary);
    }

    toString(/* radix */) {
        return this.real + ' + ' + this.imaginary + 'i';
    }
};

base.Complex128 = class Complex {

    constructor(real, imaginary) {
        this.real = real;
        this.imaginary = imaginary;
    }

    static create(real, imaginary) {
        return new base.Complex128(real, imaginary);
    }

    toString(/* radix */) {
        return this.real + ' + ' + this.imaginary + 'i';
    }
};

if (!DataView.prototype.getFloat16) {
    DataView.prototype.getFloat16 = function(byteOffset, littleEndian) {
        const value = this.getUint16(byteOffset, littleEndian);
        const e = (value & 0x7C00) >> 10;
        let f = value & 0x03FF;
        if (e == 0) {
            f = 0.00006103515625 * (f / 1024);
        } else if (e == 0x1F) {
            f = f ? NaN : Infinity;
        } else {
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

if (!DataView.prototype.getBfloat16) {
    DataView.prototype.getBfloat16 = function(byteOffset, littleEndian) {
        if (littleEndian) {
            DataView.__bfloat16_get_uint16_le[1] = this.getUint16(byteOffset, littleEndian);
            return DataView.__bfloat16_get_float32_le[0];
        }
        DataView.__bfloat16_uint16_be[0] = this.getUint16(byteOffset, littleEndian);
        return DataView.__bfloat16_get_float32_be[0];
    };
    DataView.__bfloat16_get_float32_le = new Float32Array(1);
    DataView.__bfloat16_get_float32_be = new Float32Array(1);
    DataView.__bfloat16_get_uint16_le = new Uint16Array(DataView.__bfloat16_get_float32_le.buffer, DataView.__bfloat16_get_float32_le.byteOffset, 2);
    DataView.__bfloat16_get_uint16_be = new Uint16Array(DataView.__bfloat16_get_float32_be.buffer, DataView.__bfloat16_get_float32_be.byteOffset, 2);
}

DataView.__float8e4m3_float32 = new Float32Array(1);
DataView.__float8e4m3_uint32 = new Uint32Array(DataView.__float8e4m3_float32.buffer, DataView.__float8e4m3_float32.byteOffset, 1);
DataView.prototype.getFloat8e4m3 = function(byteOffset, fn, uz) {
    const value = this.getUint8(byteOffset);
    let exponent_bias = 7;
    if (uz) {
        exponent_bias = 8;
        if (value == 0x80) {
            return NaN;
        }
    } else if (value === 255) {
        return -NaN;
    } else if (value === 0x7f) {
        return NaN;
    }
    let expo = (value & 0x78) >> 3;
    let mant = value & 0x07;
    const sign = value & 0x80;
    let res = sign << 24;
    if (expo == 0) {
        if (mant > 0) {
            expo = 0x7F - exponent_bias;
            if (mant & 0x4 == 0) {
                mant &= 0x3;
                mant <<= 1;
                expo -= 1;
            }
            if (mant & 0x4 == 0) {
                mant &= 0x3;
                mant <<= 1;
                expo -= 1;
            }
            res |= (mant & 0x3) << 21;
            res |= expo << 23;
        }
    } else {
        res |= mant << 20;
        expo += 0x7F - exponent_bias;
        res |= expo << 23;
    }
    DataView.__float8e4m3_uint32[0] = res;
    return DataView.__float8e4m3_float32[0];
};

DataView.__float8e5m2_float32 = new Float32Array(1);
DataView.__float8e5m2_uint32 = new Uint32Array(DataView.__float8e5m2_float32.buffer, DataView.__float8e5m2_float32.byteOffset, 1);
DataView.prototype.getFloat8e5m2 = function(byteOffset, fn, uz) {
    const value = this.getUint8(byteOffset);
    let exponent_bias = NaN;
    if (fn && uz) {
        if (value == 0x80) {
            return NaN;
        }
        exponent_bias = 16;
    } else if (!fn && !uz) {
        if (value >= 253 && value <= 255) {
            return -NaN;
        }
        if (value >= 126 && value <= 127) {
            return NaN;
        }
        if (value === 252) {
            return -Infinity;
        }
        if (value === 124) {
            return Infinity;
        }
        exponent_bias = 15;
    }
    let expo = (value & 0x7C) >> 2;
    let mant = value & 0x03;
    let res = (value & 0x80) << 24;
    if (expo == 0) {
        if (mant > 0) {
            expo = 0x7F - exponent_bias;
            if (mant & 0x2 == 0) {
                mant &= 0x1;
                mant <<= 1;
                expo -= 1;
            }
            res |= (mant & 0x1) << 22;
            res |= expo << 23;
        }
    } else {
        res |= mant << 21;
        expo += 0x7F - exponent_bias;
        res |= expo << 23;
    }
    DataView.__float8e5m2_uint32[0] = res;
    return DataView.__float8e5m2_float32[0];
};

DataView.prototype.getInt64 = DataView.prototype.getInt64 || function(byteOffset, littleEndian) {
    return littleEndian ?
        new base.Int64(this.getUint32(byteOffset, true), this.getUint32(byteOffset + 4, true)) :
        new base.Int64(this.getUint32(byteOffset + 4, true), this.getUint32(byteOffset, true));
};

DataView.prototype.setInt64 = DataView.prototype.setInt64 || function(byteOffset, value, littleEndian) {
    if (littleEndian) {
        this.setUint32(byteOffset, value.low, true);
        this.setUint32(byteOffset + 4, value.high, true);
    } else {
        this.setUint32(byteOffset + 4, value.low, false);
        this.setUint32(byteOffset, value.high, false);
    }
};

DataView.prototype.getIntBits = DataView.prototype.getUintBits || function(offset, bits) {
    offset = offset * bits;
    const available = (this.byteLength << 3) - offset;
    if (bits > available) {
        throw new RangeError("Invalid bit size '" + bits + "'.");
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
    return (value < (2 << (bits - 1)) ? value : (2 << bits));
};

DataView.prototype.getUint64 = DataView.prototype.getUint64 || function(byteOffset, littleEndian) {
    return littleEndian ?
        new base.Uint64(this.getUint32(byteOffset, true), this.getUint32(byteOffset + 4, true)) :
        new base.Uint64(this.getUint32(byteOffset + 4, true), this.getUint32(byteOffset, true));
};

DataView.prototype.setUint64 = DataView.prototype.setUint64 || function(byteOffset, value, littleEndian) {
    if (littleEndian) {
        this.setUint32(byteOffset, value.low, true);
        this.setUint32(byteOffset + 4, value.high, true);
    } else {
        this.setUint32(byteOffset + 4, value.low, false);
        this.setUint32(byteOffset, value.high, false);
    }
};

DataView.prototype.getUintBits = DataView.prototype.getUintBits || function(offset, bits) {
    offset = offset * bits;
    const available = (this.byteLength << 3) - offset;
    if (bits > available) {
        throw new RangeError("Invalid bit size '" + bits + "'.");
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

DataView.prototype.getComplex64 = DataView.prototype.getComplex64 || function(byteOffset, littleEndian) {
    const real = littleEndian ? this.getFloat32(byteOffset, littleEndian) : this.getFloat32(byteOffset + 4, littleEndian);
    const imaginary = littleEndian ? this.getFloat32(byteOffset + 4, littleEndian) : this.getFloat32(byteOffset, littleEndian);
    return base.Complex64.create(real, imaginary);
};

DataView.prototype.setComplex64 = DataView.prototype.setComplex64 || function(byteOffset, value, littleEndian) {
    if (littleEndian) {
        this.setFloat32(byteOffset, value.real, littleEndian);
        this.setFloat32(byteOffset + 4, value.imaginary, littleEndian);
    } else {
        this.setFloat32(byteOffset + 4, value.real, littleEndian);
        this.setFloat32(byteOffset, value.imaginary, littleEndian);
    }
};

DataView.prototype.getComplex128 = DataView.prototype.getComplex128 || function(byteOffset, littleEndian) {
    const real = littleEndian ? this.getFloat64(byteOffset, littleEndian) : this.getFloat64(byteOffset + 8, littleEndian);
    const imaginary = littleEndian ? this.getFloat64(byteOffset + 8, littleEndian) : this.getFloat64(byteOffset, littleEndian);
    return base.Complex128.create(real, imaginary);
};

DataView.prototype.setComplex128 = DataView.prototype.setComplex128 || function(byteOffset, value, littleEndian) {
    if (littleEndian) {
        this.setFloat64(byteOffset, value.real, littleEndian);
        this.setFloat64(byteOffset + 8, value.imaginary, littleEndian);
    } else {
        this.setFloat64(byteOffset + 8, value.real, littleEndian);
        this.setFloat64(byteOffset, value.imaginary, littleEndian);
    }
};

base.BinaryStream = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._length = buffer.length;
        this._position = 0;
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    stream(length) {
        const buffer = this.read(length);
        return new base.BinaryStream(buffer.slice(0));
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
        if (this._position > this._buffer.length) {
            throw new Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    peek(length) {
        if (this._position === 0 && length === undefined) {
            return this._buffer;
        }
        const position = this._position;
        this.skip(length !== undefined ? length : this._length - this._position);
        const end = this._position;
        this.seek(position);
        return this._buffer.subarray(position, end);
    }

    read(length) {
        if (this._position === 0 && length === undefined) {
            this._position = this._length;
            return this._buffer;
        }
        const position = this._position;
        this.skip(length !== undefined ? length : this._length - this._position);
        return this._buffer.subarray(position, this._position);
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._buffer[position];
    }
};

base.BinaryReader = class {

    constructor(data) {
        this._buffer = data instanceof Uint8Array ? data : data.peek();
        this._position = 0;
        this._length = this._buffer.length;
        this._view = new DataView(this._buffer.buffer, this._buffer.byteOffset, this._buffer.byteLength);
    }

    get length() {
        return this._length;
    }

    get position() {
        return this._position;
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
        if (this._position > this._length || this._position < 0) {
            throw new Error('Expected ' + (this._position - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._length) {
            throw new Error('Expected ' + (this._position - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    align(mod) {
        if (this._position % mod != 0) {
            this.skip(mod - (this._position % mod));
        }
    }

    peek(length) {
        if (this._position === 0 && length === undefined) {
            return this._buffer;
        }
        const position = this._position;
        this.skip(length !== undefined ? length : this._length - this._position);
        const end = this._position;
        this._position = position;
        return this._buffer.slice(position, end);
    }

    read(length) {
        if (this._position === 0 && length === undefined) {
            this._position = this._length;
            return this._buffer;
        }
        const position = this._position;
        this.skip(length !== undefined ? length : this._length - this._position);
        return this._buffer.slice(position, this._position);
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._buffer[position];
    }

    int8() {
        const position = this._position;
        this.skip(1);
        return this._view.getInt8(position, true);
    }

    int16() {
        const position = this._position;
        this.skip(2);
        return this._view.getInt16(position, true);
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._view.getInt32(position, true);
    }

    int64() {
        const position = this._position;
        this.skip(8);
        return this._view.getInt64(position, true).toNumber();
    }

    uint16() {
        const position = this._position;
        this.skip(2);
        return this._view.getUint16(position, true);
    }

    uint32() {
        const position = this._position;
        this.skip(4);
        return this._view.getUint32(position, true);
    }

    uint64() {
        const position = this._position;
        this.skip(8);
        const low = this._view.getUint32(position, true);
        const high = this._view.getUint32(position + 4, true);
        if (high === 0) {
            return low;
        }
        const value = (high * 4294967296) + low;
        if (Number.isSafeInteger(value)) {
            return value;
        }
        throw new Error("Unsigned 64-bit value exceeds safe integer.");
    }

    float32() {
        const position = this._position;
        this.skip(4);
        return this._view.getFloat32(position, true);
    }

    float64() {
        const position = this._position;
        this.skip(8);
        return this._view.getFloat64(position, true);
    }

    string() {
        const length = this.uint32();
        const position = this._position;
        this.skip(length);
        const data = this._buffer.subarray(position, this._position);
        this._decoder = this._decoder || new TextDecoder('utf-8');
        return this._decoder.decode(data);
    }

    boolean() {
        return this.byte() !== 0 ? true : false;
    }
};

base.Telemetry = class {

    constructor(window) {
        this._window = window;
        this._navigator = window.navigator;
        this._config = new Map();
        this._metadata = {};
        this._schema = new Map([
            [ 'protocol_version', 'v' ],
            [ 'tracking_id', 'tid' ],
            [ 'hash_info', 'gtm' ],
            [ '_page_id', '_p'],
            [ 'client_id', 'cid' ],
            [ 'language', 'ul' ],
            [ 'screen_resolution', 'sr' ],
            [ '_user_agent_architecture', 'uaa' ],
            [ '_user_agent_bitness', 'uab' ],
            [ '_user_agent_full_version_list', 'uafvl' ],
            [ '_user_agent_mobile', 'uamb' ],
            [ '_user_agent_model', 'uam' ],
            [ '_user_agent_platform', 'uap' ],
            [ '_user_agent_platform_version', 'uapv' ],
            [ '_user_agent_wow64', 'uaw' ],
            [ 'hit_count', '_s' ],
            [ 'session_id', 'sid' ],
            [ 'session_number', 'sct' ],
            [ 'session_engaged', 'seg' ],
            [ 'engagement_time_msec', '_et' ],
            [ 'page_location', 'dl' ],
            [ 'page_title', 'dt' ],
            [ 'page_referrer', 'dr' ],
            [ 'is_first_visit', '_fv' ],
            [ 'is_external_event', '_ee' ],
            [ 'is_new_to_site', '_nsi' ],
            [ 'is_session_start', '_ss' ],
            [ 'event_name', 'en' ]
        ]);
    }

    async start(measurement_id, client_id, session) {
        this._session = session && typeof session === 'string' ? session.replace(/^GS1\.1\./, '').split('.') : null;
        this._session = Array.isArray(this._session) && this._session.length >= 7 ? this._session : [ '0', '0', '0', '0', '0', '0', '0' ];
        this._session[0] = Date.now();
        this._session[1] = parseInt(this._session[1], 10) + 1;
        this._engagement_time_msec = 0;
        if (this._config.size > 0) {
            throw new Error('Invalid session state.');
        }
        this.set('protocol_version', 2);
        this.set('tracking_id', measurement_id);
        this.set('hash_info', '2oebu0');
        this.set('_page_id', Math.floor(Math.random() * 2147483648));
        client_id = client_id ? client_id.replace(/^(GA1\.\d\.)*/, '') : null;
        if (client_id && client_id.indexOf('.') !== 1) {
            this.set('client_id', client_id);
        } else {
            const random = String(Math.round(0x7FFFFFFF * Math.random()));
            const time = Date.now();
            const value = [ random, Math.round(time / 1e3) ].join('.');
            this.set('client_id', value);
            this._metadata.is_first_visit = 1;
            this._metadata.is_new_to_site = 1;
        }
        this.set('language', ((this._navigator && (this._navigator.language || this._navigator.browserLanguage)) || '').toLowerCase());
        this.set('screen_resolution', (window.screen ? window.screen.width : 0) + 'x' + (window.screen ? window.screen.height : 0));
        if (this._navigator && this._navigator.userAgentData && this._navigator.userAgentData.getHighEntropyValues) {
            const values = await this._navigator.userAgentData.getHighEntropyValues([ 'platform', 'platformVersion', 'architecture', 'model', 'uaFullVersion', 'bitness', 'fullVersionList', 'wow64' ]);
            if (values) {
                this.set('_user_agent_architecture', values.architecture);
                this.set('_user_agent_bitness', values.bitness);
                this.set('_user_agent_full_version_list', Array.isArray(values.fullVersionList) ? values.fullVersionList.map((h) => encodeURIComponent(h.brand || '') + ';' + encodeURIComponent(h.version || '')).join('|') : '');
                this.set('_user_agent_mobile', values.mobile ? 1 : 0);
                this.set('_user_agent_model', values.model);
                this.set('_user_agent_platform', values.platform);
                this.set('_user_agent_platform_version', values.platformVersion);
                this.set('_user_agent_wow64', values.wow64 ? 1 : 0);
            }
        }
        this.set('hit_count', 1);
        this.set('session_id', this._session[0]);
        this.set('session_number', this._session[1]);
        this.set('session_engaged', 0);
        this._metadata.is_session_start = 1;
        this._metadata.is_external_event = 1;
        window.addEventListener('focus', () => this._update(true, undefined, undefined));
        window.addEventListener('blur', () => this._update(false, undefined, undefined));
        window.addEventListener('pageshow', () => this._update(undefined, true, undefined));
        window.addEventListener('pagehide', () => this._update(undefined, false, undefined));
        window.addEventListener('visibilitychange', () => this._update(undefined, undefined, window.document.visibilityState !== 'hidden'));
        window.addEventListener('beforeunload', () => this._update() && this.send('user_engagement', {}));
    }

    get session() {
        return this._session ? this._session.join('.') : null;
    }

    set(name, value) {
        const key = this._schema.get(name);
        if (value !== undefined && value !== null) {
            this._config.set(key, value);
        } else if (this._config.has(key)) {
            this._config.delete(key);
        }
        this._cache = null;
    }

    get(name) {
        const key = this._schema.get(name);
        return this._config.get(key);
    }

    send(name, params) {
        if (this._session) {
            try {
                params = Object.assign({ event_name: name }, this._metadata, /* { debug_mode: true },*/ params);
                this._metadata = {};
                this._update() && (params.engagement_time_msec = this._engagement_time_msec) && (this._engagement_time_msec = 0);
                const build = (entires) => entires.map((entry) => entry[0] + '=' + encodeURIComponent(entry[1])).join('&');
                this._cache = this._cache || build(Array.from(this._config));
                const key = (name, value) => this._schema.get(name) || ('number' === typeof value && !isNaN(value) ? 'epn.' : 'ep.') + name;
                const body = build(Object.entries(params).map((entry) => [ key(entry[0], entry[1]), entry[1] ]));
                const url = 'https://analytics.google.com/g/collect?' + this._cache;
                this._navigator.sendBeacon(url, body);
                this._session[2] = this.get('session_engaged') || '0';
                this.set('hit_count', this.get('hit_count') + 1);
            } catch (e) {
                // continue regardless of error
            }
        }
    }

    _update(focused, page, visible) {
        this._focused = focused === true || focused === false ? focused : this._focused;
        this._page = page === true || page === false ? page : this._page;
        this._visible = visible === true || visible === false ? visible : this._visible;
        const time = Date.now();
        if (this._start_time) {
            this._engagement_time_msec += (time - this._start_time);
            this._start_time = 0;
        }
        if (this._focused !== false && this._page !== false && this._visible !== false) {
            this._start_time = time;
        }
        return this._engagement_time_msec > 20;
    }
};

base.Metadata = class {

    get extensions() {
        return [
            'onnx', 'tflite', 'pb', 'pt', 'pth', 'h5', 'pbtxt', 'prototxt', 'caffemodel', 'mlmodel', 'mlpackage',
            'model', 'json', 'xml', 'cfg',
            'ort',
            'dnn', 'cmf',
            'hd5', 'hdf5', 'keras',
            'tfl', 'circle', 'lite',
            'mlnet', 'mar',  'meta', 'nn', 'ngf', 'hn', 'har',
            'param', 'params',
            'paddle', 'pdiparams', 'pdmodel', 'pdopt', 'pdparams', 'nb',
            'pkl', 'joblib', 'safetensors',
            'ptl', 't7',
            'dlc', 'uff', 'armnn',
            'mnn', 'ms', 'ncnn', 'om', 'tm', 'mge', 'tmfile', 'tnnproto', 'xmodel', 'kmodel', 'rknn',
            'tar', 'zip'
        ];
    }
};

if (typeof window !== 'undefined' && typeof window.Long != 'undefined') {
    window.long = { Long: window.Long };
    window.Int64 = base.Int64;
    window.Uint64 = base.Uint64;
}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Int64 = base.Int64;
    module.exports.Uint64 = base.Uint64;
    module.exports.Complex64 = base.Complex64;
    module.exports.Complex128 = base.Complex128;
    module.exports.BinaryStream = base.BinaryStream;
    module.exports.BinaryReader = base.BinaryReader;
    module.exports.Telemetry = base.Telemetry;
    module.exports.Metadata = base.Metadata;
}
