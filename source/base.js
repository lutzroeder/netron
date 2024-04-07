
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

    align(mod) {
        const remainder = this.position % mod;
        if (remainder !== 0) {
            this.skip(mod - remainder);
        }
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

    align(mod) {
        const remainder = this.position % mod;
        if (remainder !== 0) {
            this.skip(mod - remainder);
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
            'mlnet', 'mar', 'maxviz', 'meta', 'nn', 'ngf', 'hn', 'har',
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

export const Complex64 = base.Complex64;
export const Complex128 = base.Complex128;
export const BinaryStream = base.BinaryStream;
export const BinaryReader = base.BinaryReader;
export const Telemetry = base.Telemetry;
export const Metadata = base.Metadata;
