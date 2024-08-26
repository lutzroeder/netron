
const base = {};

base.Complex64 = class Complex {

    constructor(real, imaginary) {
        this.real = real;
        this.imaginary = imaginary;
    }

    toString(/* radix */) {
        return `${this.real} + ${this.imaginary}i`;
    }
};

base.Complex128 = class Complex {

    constructor(real, imaginary) {
        this.real = real;
        this.imaginary = imaginary;
    }

    toString(/* radix */) {
        return `${this.real} + ${this.imaginary}i`;
    }
};

/* eslint-disable no-extend-native */

BigInt.prototype.toNumber = function() {
    if (this > Number.MAX_SAFE_INTEGER || this < Number.MIN_SAFE_INTEGER) {
        throw new Error('64-bit value exceeds safe integer.');
    }
    return Number(this);
};

if (!DataView.prototype.getFloat16) {
    DataView.prototype.getFloat16 = function(byteOffset, littleEndian) {
        const value = this.getUint16(byteOffset, littleEndian);
        const e = (value & 0x7C00) >> 10;
        let f = value & 0x03FF;
        if (e === 0) {
            f = 0.00006103515625 * (f / 1024);
        } else if (e === 0x1F) {
            f = f ? NaN : Infinity;
        } else {
            f = DataView.__float16_pow[e] * (1 + (f / 1024));
        }
        return value & 0x8000 ? -f : f;
    };
    DataView.__float16_pow = {
        1: 1 / 16384, 2: 1 / 8192, 3: 1 / 4096, 4: 1 / 2048, 5: 1 / 1024, 6: 1 / 512, 7: 1 / 256, 8: 1 / 128,
        9: 1 / 64, 10: 1 / 32, 11: 1 / 16, 12: 1 / 8, 13: 1 / 4, 14: 1 / 2, 15: 1, 16: 2,
        17: 4, 18: 8, 19: 16, 20: 32, 21: 64, 22: 128, 23: 256, 24: 512,
        25: 1024, 26: 2048, 27: 4096, 28: 8192, 29: 16384, 30: 32768, 31: 65536
    };
}

if (!DataView.prototype.setFloat16) {
    DataView.prototype.setFloat16 = function(byteOffset, value, littleEndian) {
        DataView.__float16_float[0] = value;
        [value] = DataView.__float16_int;
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

DataView.__float4e2m1_float32 = new Float32Array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]);
DataView.prototype.getFloat4e2m1 = function(byteOffset) {
    let value = this.getUint8(byteOffset >> 1);
    value = byteOffset & 1 ? value >> 4 : value & 0x0F;
    return DataView.__float4e2m1_float32[value];
};

DataView.__float8e4m3_float32 = new Float32Array(1);
DataView.__float8e4m3_uint32 = new Uint32Array(DataView.__float8e4m3_float32.buffer, DataView.__float8e4m3_float32.byteOffset, 1);
DataView.prototype.getFloat8e4m3 = function(byteOffset, fn, uz) {
    const value = this.getUint8(byteOffset);
    let exponent_bias = 7;
    if (uz) {
        exponent_bias = 8;
        if (value === 0x80) {
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
    if (expo === 0) {
        if (mant > 0) {
            expo = 0x7F - exponent_bias;
            if (mant & 0x4 === 0) {
                mant &= 0x3;
                mant <<= 1;
                expo -= 1;
            }
            if (mant & 0x4 === 0) {
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
        if (value === 0x80) {
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
    if (expo === 0) {
        if (mant > 0) {
            expo = 0x7F - exponent_bias;
            if (mant & 0x2 === 0) {
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

DataView.prototype.getIntBits = DataView.prototype.getUintBits || function(offset, bits, littleEndian) {
    offset *= bits;
    const position = Math.floor(offset / 8);
    const remainder = offset % 8;
    let value = 0;
    if ((remainder + bits) <= 8) {
        value = littleEndian ? this.getUint8(position) >> remainder : this.getUint8(position) >> (8 - remainder - bits);
    } else {
        value = littleEndian ? this.getUint16(position, true) >> remainder : this.getUint16(position, false) >> (16 - remainder - bits);
    }
    value &= (1 << bits) - 1;
    if (value & (1 << (bits - 1))) {
        value -= 1 << bits;
    }
    return value;
};

DataView.prototype.getUintBits = DataView.prototype.getUintBits || function(offset, bits, littleEndian) {
    offset *= bits;
    const position = Math.floor(offset / 8);
    const remainder = offset % 8;
    let value = 0;
    if ((remainder + bits) <= 8) {
        value = littleEndian ? this.getUint8(position) >> remainder : this.getUint8(position) >> (8 - remainder - bits);
    } else {
        value = littleEndian ? this.getUint16(position, true) >> remainder : this.getUint16(position, false) >> (16 - remainder - bits);
    }
    return value & ((1 << bits) - 1);
};

DataView.prototype.getComplex64 = DataView.prototype.getComplex64 || function(byteOffset, littleEndian) {
    const real = littleEndian ? this.getFloat32(byteOffset, littleEndian) : this.getFloat32(byteOffset + 4, littleEndian);
    const imaginary = littleEndian ? this.getFloat32(byteOffset + 4, littleEndian) : this.getFloat32(byteOffset, littleEndian);
    return new base.Complex64(real, imaginary);
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
    return new base.Complex128(real, imaginary);
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

/* eslint-enable no-extend-native */

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
            throw new Error(`Expected ${this._position - this._buffer.length} more bytes. The file might be corrupted. Unexpected end of file.`);
        }
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new Error(`Expected ${this._position - this._buffer.length} more bytes. The file might be corrupted. Unexpected end of file.`);
        }
    }

    peek(length) {
        if (this._position === 0 && length === undefined) {
            return this._buffer;
        }
        const position = this._position;
        this.skip(length === undefined ? this._length - this._position : length);
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
        this.skip(length === undefined ? this._length - this._position : length);
        return this._buffer.subarray(position, this._position);
    }
};

base.BinaryReader = class {

    static open(data, littleEndian) {
        if (data instanceof Uint8Array || data.length <= 0x20000000) {
            return new base.BufferReader(data, littleEndian);
        }
        return new base.StreamReader(data, littleEndian);
    }
};

base.BufferReader = class {

    constructor(data, littleEndian) {
        this._buffer = data instanceof Uint8Array ? data : data.peek();
        this._littleEndian = littleEndian !== false;
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
            throw new Error(`Expected ${this._position - this._length} more bytes. The file might be corrupted. Unexpected end of file.`);
        }
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._length) {
            throw new Error(`Expected ${this._position - this._length} more bytes. The file might be corrupted. Unexpected end of file.`);
        }
    }

    align(size) {
        const remainder = this.position % size;
        if (remainder !== 0) {
            this.skip(size - remainder);
        }
    }

    stream(length) {
        const buffer = this.read(length);
        return new base.BinaryStream(buffer);
    }

    peek(length) {
        if (this._position === 0 && length === undefined) {
            return this._buffer;
        }
        const position = this._position;
        this.skip(length === undefined ? this._length - this._position : length);
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
        this.skip(length === undefined ? this._length - this._position : length);
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
        return this._view.getInt8(position, this._littleEndian);
    }

    int16() {
        const position = this._position;
        this.skip(2);
        return this._view.getInt16(position, this._littleEndian);
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._view.getInt32(position, this._littleEndian);
    }

    int64() {
        const position = this._position;
        this.skip(8);
        return this._view.getBigInt64(position, this._littleEndian);
    }

    uint16() {
        const position = this._position;
        this.skip(2);
        return this._view.getUint16(position, this._littleEndian);
    }

    uint32() {
        const position = this._position;
        this.skip(4);
        return this._view.getUint32(position, this._littleEndian);
    }

    uint64() {
        const position = this._position;
        this.skip(8);
        return this._view.getBigUint64(position, this._littleEndian);
    }

    float32() {
        const position = this._position;
        this.skip(4);
        return this._view.getFloat32(position, this._littleEndian);
    }

    float64() {
        const position = this._position;
        this.skip(8);
        return this._view.getFloat64(position, this._littleEndian);
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
        return this.byte() === 0 ? false : true;
    }
};

base.StreamReader = class {

    constructor(stream, littleEndian) {
        this._stream = stream;
        this._littleEndian = littleEndian !== false;
        this._buffer = new Uint8Array(8);
        this._view = new DataView(this._buffer.buffer, this._buffer.byteOffset, this._buffer.byteLength);
    }

    get position() {
        return this._stream.position;
    }

    get length() {
        return this._stream.length;
    }

    seek(position) {
        this._stream.seek(position);
    }

    skip(offset) {
        this._stream.skip(offset);
    }

    align(size) {
        const remainder = this.position % size;
        if (remainder !== 0) {
            this.skip(size - remainder);
        }
    }

    stream(length) {
        return this._stream.stream(length);
    }

    peek(length) {
        return this._stream.peek(length).slice(0);
    }

    read(length) {
        return this._stream.read(length).slice(0);
    }

    byte() {
        return this._stream.read(1)[0];
    }

    int8() {
        const buffer = this._stream.read(1);
        this._buffer.set(buffer, 0);
        return this._view.getInt8(0);
    }

    int16() {
        const buffer = this._stream.read(2);
        this._buffer.set(buffer, 0);
        return this._view.getInt16(0, this._littleEndian);
    }

    int32() {
        const buffer = this._stream.read(4);
        this._buffer.set(buffer, 0);
        return this._view.getInt32(0, this._littleEndian);
    }

    int64() {
        const buffer = this._stream.read(8);
        this._buffer.set(buffer, 0);
        return this._view.getBigInt64(0, this._littleEndian);
    }

    uint16() {
        const buffer = this._stream.read(2);
        this._buffer.set(buffer, 0);
        return this._view.getUint16(0, this._littleEndian);
    }

    uint32() {
        const buffer = this._stream.read(4);
        this._buffer.set(buffer, 0);
        return this._view.getUint32(0, this._littleEndian);
    }

    uint64() {
        const buffer = this._stream.read(8);
        this._buffer.set(buffer, 0);
        return this._view.getBigUint64(0, this._littleEndian);
    }

    float32() {
        const buffer = this._stream.read(4);
        this._buffer.set(buffer, 0);
        return this._view.getFloat32(0, this._littleEndian);
    }

    float64() {
        const buffer = this._stream.read(8);
        this._buffer.set(buffer, 0);
        return this._view.getFloat64(0, this._littleEndian);
    }

    boolean() {
        return this.byte() === 0 ? false : true;
    }
};

base.Tensor = class {

    constructor(tensor) {
        this._tensor = tensor;
        this.name = tensor.name || '';
        this.encoding = tensor.encoding;
        this.encoding = this.encoding === '' || this.encoding === undefined ? '<' : this.encoding;
        this.type = tensor.type;
        this.layout = tensor.type.layout;
        this.stride = tensor.stride;
        base.Tensor._dataTypes = base.Tensor._dataTypes || new Map([
            ['boolean', 1],
            ['qint8', 1], ['qint16', 2], ['qint32', 4],
            ['quint8', 1], ['quint16', 2], ['quint32', 4],
            ['xint8', 1],
            ['int8', 1], ['int16', 2], ['int32', 4], ['int64', 8],
            ['uint8', 1], ['uint16', 2], ['uint32', 4,], ['uint64', 8],
            ['float16', 2], ['float32', 4], ['float64', 8], ['bfloat16', 2],
            ['complex64', 8], ['complex128', 16],
            ['float8e4m3fn', 1], ['float8e4m3fnuz', 1], ['float8e5m2', 1], ['float8e5m2fnuz', 1]
        ]);
    }

    get values() {
        this._read();
        return this._values;
    }

    get indices() {
        this._read();
        return this._indices;
    }

    get data() {
        this._read();
        return this._data;
    }

    get metrics() {
        return this._tensor.metrics;
    }

    get empty() {
        switch (this.layout) {
            case 'sparse':
            case 'sparse.coo': {
                return !this.values || !this.indices || this.values.values === null || this.values.values.length === 0;
            }
            default: {
                switch (this.encoding) {
                    case '<':
                    case '>':
                        return !(Array.isArray(this.data) || this.data instanceof Uint8Array || this.data instanceof Int8Array) || this.data.length === 0;
                    case '|':
                        return !(Array.isArray(this.values) || ArrayBuffer.isView(this.values)) || this.values.length === 0;
                    default:
                        throw new Error(`Unsupported tensor encoding '${this.encoding}'.`);
                }
            }
        }
    }

    get value() {
        const context = this._context();
        context.limit = Number.MAX_SAFE_INTEGER;
        switch (context.encoding) {
            case '<':
            case '>': {
                return this._decodeData(context, 0, 0);
            }
            case '|': {
                return this._decodeValues(context, 0, 0);
            }
            default: {
                throw new Error(`Unsupported tensor encoding '${context.encoding}'.`);
            }
        }
    }

    toString() {
        const context = this._context();
        context.limit = 10000;
        switch (context.encoding) {
            case '<':
            case '>': {
                const value = this._decodeData(context, 0, 0);
                return base.Tensor._stringify(value, '', '    ');
            }
            case '|': {
                const value = this._decodeValues(context, 0, 0);
                return base.Tensor._stringify(value, '', '    ');
            }
            default: {
                throw new Error(`Unsupported tensor encoding '${context.encoding}'.`);
            }
        }
    }

    _context() {
        this._read();
        if (this.encoding !== '<' && this.encoding !== '>' && this.encoding !== '|') {
            throw new Error(`Tensor encoding '${this.encoding}' is not supported.`);
        }
        if (this.layout && (this.layout !== 'sparse' && this.layout !== 'sparse.coo')) {
            throw new Error(`Tensor layout '${this.layout}' is not supported.`);
        }
        const dataType = this.type.dataType;
        const context = {};
        context.encoding = this.encoding;
        context.dimensions = this.type.shape.dimensions.map((value) => typeof value === 'bigint' ? value.toNumber() : value);
        context.dataType = dataType;
        const shape = context.dimensions;
        context.stride = this.stride;
        if (!Array.isArray(context.stride)) {
            context.stride = new Array(shape.length);
            let value = 1;
            for (let i = shape.length - 1; i >= 0; i--) {
                context.stride[i] = value;
                value *= shape[i];
            }
        }
        switch (this.layout) {
            case 'sparse': {
                const indices = new base.Tensor(this.indices).value;
                const values = new base.Tensor(this.values).value;
                context.data = this._decodeSparse(dataType, context.dimensions, indices, values);
                context.encoding = '|';
                break;
            }
            case 'sparse.coo': {
                const values = new base.Tensor(this.values).value;
                const data = new base.Tensor(this.indices).value;
                const dimensions = context.dimensions.length;
                let stride = 1;
                const strides = context.dimensions.slice().reverse().map((dim) => {
                    const value = stride;
                    stride *= dim;
                    return value;
                }).reverse();
                const indices = new Uint32Array(values.length);
                for (let i = 0; i < dimensions; i++) {
                    const stride = strides[i];
                    const dimension = data[i];
                    for (let i = 0; i < indices.length; i++) {
                        indices[i] += dimension[i].toNumber() * stride;
                    }
                }
                context.data = this._decodeSparse(dataType, context.dimensions, indices, values);
                context.encoding = '|';
                break;
            }
            default: {
                switch (this.encoding) {
                    case '<':
                    case '>': {
                        context.data = (this.data instanceof Uint8Array || this.data instanceof Int8Array) ? this.data : this.data.peek();
                        context.view = new DataView(context.data.buffer, context.data.byteOffset, context.data.byteLength);
                        if (base.Tensor._dataTypes.has(dataType)) {
                            const itemsize = base.Tensor._dataTypes.get(dataType);
                            const length = context.data.length;
                            const stride = context.stride;
                            if (length < (itemsize * shape.reduce((a, v) => a * v, 1))) {
                                const max = stride.reduce((a, v, i) => v > stride[i] ? i : a, 0);
                                if (length !== (itemsize * stride[max] * shape[max])) {
                                    throw new Error('Invalid tensor data size.');
                                }
                            }
                            context.itemsize = itemsize;
                            context.stride = stride.map((v) => v * itemsize);
                        } else if (dataType.startsWith('uint') && !isNaN(parseInt(dataType.substring(4), 10))) {
                            context.dataType = 'uint';
                            context.bits = parseInt(dataType.substring(4), 10);
                            context.itemsize = 1;
                        } else if (dataType.startsWith('int') && !isNaN(parseInt(dataType.substring(3), 10))) {
                            context.dataType = 'int';
                            context.bits = parseInt(dataType.substring(3), 10);
                            context.itemsize = 1;
                        } else if (dataType === 'float4e2m1') {
                            context.dataType = 'float4e2m1';
                            context.bits = 4;
                            context.itemsize = 1;
                        } else {
                            throw new Error(`Tensor data type '${dataType}' is not implemented.`);
                        }
                        break;
                    }
                    case '|': {
                        context.data = this.values;
                        if (!base.Tensor._dataTypes.has(dataType) && dataType !== 'string' && dataType !== 'object' && dataType !== 'void') {
                            throw new Error(`Tensor data type '${dataType}' is not implemented.`);
                        }
                        const size = context.dimensions.reduce((a, v) => a * v, 1);
                        if (size !== this.values.length) {
                            throw new Error('Invalid tensor data length.');
                        }
                        break;
                    }
                    default: {
                        throw new base.Tensor(`Unsupported tensor encoding '${this.encoding}'.`);
                    }
                }
            }
        }
        context.index = 0;
        context.count = 0;
        return context;
    }

    _decodeSparse(dataType, dimensions, indices, values) {
        const size = dimensions.reduce((a, b) => a * b, 1);
        const array = new Array(size);
        switch (dataType) {
            case 'boolean':
                array.fill(false);
                break;
            default:
                array.fill(0);
                break;
        }
        if (indices.length > 0) {
            if (Object.prototype.hasOwnProperty.call(indices[0], 'low')) {
                for (let i = 0; i < indices.length; i++) {
                    const index = indices[i].toNumber();
                    array[index] = values[i];
                }
            } else {
                for (let i = 0; i < indices.length; i++) {
                    array[indices[i]] = values[i];
                }
            }
        }
        return array;
    }

    _decodeData(context, dimension, offset) {
        const results = [];
        const shape = context.dimensions.length === 0 ? [1] : context.dimensions;
        const size = shape[dimension];
        const dataType = context.dataType;
        const view = context.view;
        const stride = context.stride[dimension];
        if (dimension === shape.length - 1) {
            const ellipsis = (context.count + size) > context.limit;
            const length = ellipsis ? context.limit - context.count : size;
            const max = offset + (length * context.itemsize);
            switch (dataType) {
                case 'boolean':
                    for (; offset < max; offset += stride) {
                        results.push(view.getUint8(offset) !== 0);
                    }
                    break;
                case 'qint8':
                case 'xint8':
                case 'int8':
                    for (; offset < max; offset += stride) {
                        results.push(view.getInt8(offset));
                    }
                    break;
                case 'qint16':
                case 'int16':
                    for (; offset < max; offset += stride) {
                        results.push(view.getInt16(offset, this._littleEndian));
                    }
                    break;
                case 'qint32':
                case 'int32':
                    for (; offset < max; offset += stride) {
                        results.push(view.getInt32(offset, this._littleEndian));
                    }
                    break;
                case 'int64':
                    for (; offset < max; offset += stride) {
                        results.push(view.getBigInt64(offset, this._littleEndian));
                    }
                    break;
                case 'int':
                    for (; offset < max; offset += stride) {
                        results.push(view.getIntBits(offset, context.bits, this._littleEndian));
                    }
                    break;
                case 'quint8':
                case 'uint8':
                    for (; offset < max; offset += stride) {
                        results.push(view.getUint8(offset));
                    }
                    break;
                case 'quint16':
                case 'uint16':
                    for (; offset < max; offset += stride) {
                        results.push(view.getUint16(offset, true));
                    }
                    break;
                case 'quint32':
                case 'uint32':
                    for (; offset < max; offset += stride) {
                        results.push(view.getUint32(offset, true));
                    }
                    break;
                case 'uint64':
                    for (; offset < max; offset += stride) {
                        results.push(view.getBigUint64(offset, true));
                    }
                    break;
                case 'uint':
                    for (; offset < max; offset += stride) {
                        results.push(view.getUintBits(offset, context.bits, this._littleEndian));
                    }
                    break;
                case 'float16':
                    for (; offset < max; offset += stride) {
                        results.push(view.getFloat16(offset, this._littleEndian));
                    }
                    break;
                case 'float32':
                    for (; offset < max; offset += stride) {
                        results.push(view.getFloat32(offset, this._littleEndian));
                    }
                    break;
                case 'float64':
                    for (; offset < max; offset += stride) {
                        results.push(view.getFloat64(offset, this._littleEndian));
                    }
                    break;
                case 'bfloat16':
                    for (; offset < max; offset += stride) {
                        results.push(view.getBfloat16(offset, this._littleEndian));
                    }
                    break;
                case 'complex64':
                    for (; offset < max; offset += stride) {
                        results.push(view.getComplex64(offset, this._littleEndian));
                    }
                    break;
                case 'complex128':
                    for (; offset < max; offset += stride) {
                        results.push(view.getComplex128(offset, this._littleEndian));
                    }
                    break;
                case 'float4e2m1':
                    for (; offset < max; offset += stride) {
                        results.push(view.getFloat4e2m1(offset));
                    }
                    break;
                case 'float8e4m3fn':
                    for (; offset < max; offset += stride) {
                        results.push(view.getFloat8e4m3(offset, true, false));
                    }
                    break;
                case 'float8e4m3fnuz':
                    for (; offset < max; offset += stride) {
                        results.push(view.getFloat8e4m3(offset, true, true));
                    }
                    break;
                case 'float8e5m2':
                    for (; offset < max; offset += stride) {
                        results.push(view.getFloat8e5m2(offset, false, false));
                    }
                    break;
                case 'float8e5m2fnuz':
                    for (; offset < max; offset += stride) {
                        results.push(view.getFloat8e5m2(offset, true, true));
                    }
                    break;
                default:
                    throw new Error(`Unsupported tensor data type '${dataType}'.`);
            }
            context.count += length;
            if (ellipsis) {
                results.push('...');
            }
        } else {
            for (let j = 0; j < size; j++) {
                if (context.count >= context.limit) {
                    results.push('...');
                    return results;
                }
                const nextOffset = offset + (j * stride);
                results.push(this._decodeData(context, dimension + 1, nextOffset));
            }
        }
        if (context.dimensions.length === 0) {
            return results[0];
        }
        return results;
    }

    _decodeValues(context, dimension, position) {
        const results = [];
        const shape = (context.dimensions.length === 0) ? [1] : context.dimensions;
        const size = shape[dimension];
        const dataType = context.dataType;
        const stride = context.stride[dimension];
        if (dimension === shape.length - 1) {
            const ellipsis = (context.count + size) > context.limit;
            const length = ellipsis ? context.limit - context.count : size;
            const data = context.data;
            for (let i = 0; i < length; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (dataType) {
                    case 'boolean':
                        results.push(data[position] === 0 ? false : true);
                        break;
                    default:
                        results.push(data[position]);
                        break;
                }
                position += stride;
                context.count++;
            }
        } else {
            for (let i = 0; i < size; i++) {
                if (context.count >= context.limit) {
                    results.push('...');
                    return results;
                }
                const nextPosition = position + (i * stride);
                results.push(this._decodeValues(context, dimension + 1, nextPosition));
            }
        }
        if (context.dimensions.length === 0) {
            return results[0];
        }
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            const length = value.length;
            if (length > 0) {
                const array = new Array(length);
                const space = indentation + indent;
                for (let i = 0; i < length; i++) {
                    array[i] = base.Tensor._stringify(value[i], space, indent);
                }
                return `${indentation}[\n${array.join(',\n')}\n${indentation}]`;
            }
            return `${indentation}[\n${indentation}]`;
        }
        if (value === null) {
            return `${indentation}null`;
        }
        switch (typeof value) {
            case 'boolean':
            case 'number':
            case 'bigint':
                return `${indentation}${value}`;
            case 'string':
                return `${indentation}"${value}"`;
            default:
                if (value instanceof Uint8Array) {
                    let content = '';
                    for (let i = 0; i < value.length; i++) {
                        const x = value[i];
                        content += x >= 32 && x <= 126 ? String.fromCharCode(x) : `\\x${x.toString(16).padStart(2, '0')}`;
                    }
                    return `${indentation}"${content}"`;
                }
                if (value && value.toString) {
                    return `${indentation}${value.toString()}`;
                }
                return `${indentation}(undefined)`;
        }
    }

    _read() {
        if (this._values === undefined) {
            this._values = null;
            switch (this.encoding) {
                case undefined:
                case '<': {
                    this._data = this._tensor.values;
                    this._littleEndian = true;
                    break;
                }
                case '>': {
                    this._data = this._tensor.values;
                    this._littleEndian = false;
                    break;
                }
                case '|': {
                    this._values = this._tensor.values;
                    break;
                }
                default: {
                    throw new Error(`Unsupported tensor encoding '${this._encoding}'.`);
                }
            }
            switch (this.layout) {
                case 'sparse':
                case 'sparse.coo': {
                    this._indices = this._tensor.indices;
                    this._values = this._tensor.values;
                    break;
                }
                default: {
                    break;
                }
            }
        }
    }
};

base.Telemetry = class {

    constructor(window) {
        this._window = window;
        this._navigator = window.navigator;
        this._config = new Map();
        this._metadata = {};
        this._schema = new Map([
            ['protocol_version', 'v'],
            ['tracking_id', 'tid'],
            ['hash_info', 'gtm'],
            ['_page_id', '_p'],
            ['client_id', 'cid'],
            ['language', 'ul'],
            ['screen_resolution', 'sr'],
            ['_user_agent_architecture', 'uaa'],
            ['_user_agent_bitness', 'uab'],
            ['_user_agent_full_version_list', 'uafvl'],
            ['_user_agent_mobile', 'uamb'],
            ['_user_agent_model', 'uam'],
            ['_user_agent_platform', 'uap'],
            ['_user_agent_platform_version', 'uapv'],
            ['_user_agent_wow64', 'uaw'],
            ['hit_count', '_s'],
            ['session_id', 'sid'],
            ['session_number', 'sct'],
            ['session_engaged', 'seg'],
            ['engagement_time_msec', '_et'],
            ['page_location', 'dl'],
            ['page_title', 'dt'],
            ['page_referrer', 'dr'],
            ['is_first_visit', '_fv'],
            ['is_external_event', '_ee'],
            ['is_new_to_site', '_nsi'],
            ['is_session_start', '_ss'],
            ['event_name', 'en']
        ]);
    }

    async start(measurement_id, client_id, session) {
        this._session = session && typeof session === 'string' ? session.replace(/^GS1\.1\./, '').split('.') : null;
        this._session = Array.isArray(this._session) && this._session.length >= 7 ? this._session : ['0', '0', '0', '0', '0', '0', '0'];
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
            const value = [random, Math.round(time / 1e3)].join('.');
            this.set('client_id', value);
            this._metadata.is_first_visit = 1;
            this._metadata.is_new_to_site = 1;
        }
        this.set('language', ((this._navigator && (this._navigator.language || this._navigator.browserLanguage)) || '').toLowerCase());
        this.set('screen_resolution', `${window.screen ? window.screen.width : 0}x${window.screen ? window.screen.height : 0}`);
        if (this._navigator && this._navigator.userAgentData && this._navigator.userAgentData.getHighEntropyValues) {
            const values = await this._navigator.userAgentData.getHighEntropyValues(['platform', 'platformVersion', 'architecture', 'model', 'uaFullVersion', 'bitness', 'fullVersionList', 'wow64']);
            if (values) {
                this.set('_user_agent_architecture', values.architecture);
                this.set('_user_agent_bitness', values.bitness);
                this.set('_user_agent_full_version_list', Array.isArray(values.fullVersionList) ? values.fullVersionList.map((h) => `${encodeURIComponent(h.brand || '')};${encodeURIComponent(h.version || '')}`).join('|') : '');
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
                params = { event_name: name, ...this._metadata, ...params };
                this._metadata = {};
                if (this._update()) {
                    params.engagement_time_msec = this._engagement_time_msec;
                    this._engagement_time_msec = 0;
                }
                const build = (entries) => entries.map(([name, value]) => `${name}=${encodeURIComponent(value)}`).join('&');
                this._cache = this._cache || build(Array.from(this._config));
                const key = (name, value) => this._schema.get(name) || (typeof value === 'number' && !isNaN(value) ? 'epn.' : 'ep.') + name;
                const body = build(Object.entries(params).map(([name, value]) => [key(name, value), value]));
                const url = `https://analytics.google.com/g/collect?${this._cache}`;
                this._navigator.sendBeacon(url, body);
                this._session[2] = this.get('session_engaged') || '0';
                this.set('hit_count', this.get('hit_count') + 1);
            } catch {
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
            'onnx', 'tflite', 'pb', 'pt', 'pt2', 'pth', 'h5', 'pbtxt', 'prototxt', 'caffemodel', 'mlmodel', 'mlpackage',
            'model', 'json', 'xml', 'cfg', 'weights', 'bin',
            'ort',
            'dnn', 'cmf',
            'gguf',
            'hd5', 'hdf5', 'keras',
            'tfl', 'circle', 'lite',
            'mlnet', 'mar', 'maxviz', 'meta', 'nn', 'ngf', 'hn',
            'param', 'params',
            'paddle', 'pdiparams', 'pdmodel', 'pdopt', 'pdparams', 'nb',
            'pkl', 'pickle', 'joblib', 'safetensors',
            'ptl', 't7',
            'dlc', 'uff', 'armnn',
            'mnn', 'ms', 'ncnn', 'om', 'tm', 'mge', 'tmfile', 'tnnproto', 'xmodel', 'kmodel', 'rknn',
            'tar', 'zip'
        ];
    }
};

export const Complex64 = base.Complex64;
export const Complex128 = base.Complex128;
export const BinaryStream = base.BinaryStream;
export const BinaryReader = base.BinaryReader;
export const Tensor = base.Tensor;
export const Telemetry = base.Telemetry;
export const Metadata = base.Metadata;
