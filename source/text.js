
const text = {};

text.Decoder = class {

    static open(data, encoding) {
        if (typeof data === 'string') {
            return new text.Decoder.String(data);
        }
        const assert = (encoding, condition) => {
            if (encoding && encoding !== condition) {
                throw new text.Error(`Invalid encoding '${encoding}'.`);
            }
        };
        const buffer = data instanceof Uint8Array ? data : data.peek();
        const length = buffer.length;
        if (length >= 3 && buffer[0] === 0xef && buffer[1] === 0xbb && buffer[2] === 0xbf) {
            assert(encoding, 'utf-8');
            return new text.Decoder.Utf8(buffer, 3, true);
        }
        if (length >= 2 && buffer[0] === 0xff && buffer[1] === 0xfe) {
            assert(encoding, 'utf-16');
            return new text.Decoder.Utf16LE(buffer, 2);
        }
        if (length >= 2 && buffer[0] === 0xfe && buffer[1] === 0xff) {
            assert(encoding, 'utf-16');
            return new text.Decoder.Utf16BE(buffer, 2);
        }
        if (length >= 4 && buffer[0] === 0x00 && buffer[1] === 0x00 && buffer[2] === 0xfe && buffer[3] === 0xff) {
            assert(encoding, 'utf-32');
            return new text.Decoder.Utf32LE(buffer, 2);
        }
        if (length >= 4 && buffer[0] === 0xff && buffer[1] === 0xfe && buffer[2] === 0x00 && buffer[3] === 0x00) {
            assert(encoding, 'utf-32');
            return new text.Decoder.Utf32BE(buffer, 2);
        }
        if (length >= 5 && buffer[0] === 0x2B && buffer[1] === 0x2F && buffer[2] === 0x76 && buffer[3] === 0x38 && buffer[4] === 0x2D) {
            throw new text.Error("Unsupported UTF-7 encoding.");
        }
        if (length >= 4 && buffer[0] === 0x2B && buffer[1] === 0x2F && buffer[2] === 0x76 && (buffer[3] === 0x38 || buffer[3] === 0x39 || buffer[3] === 0x2B || buffer[3] === 0x2F)) {
            throw new text.Error("Unsupported UTF-7 encoding.");
        }
        if (length >= 4 && buffer[0] === 0x84 && buffer[1] === 0x31 && buffer[2] === 0x95 && buffer[3] === 0x33) {
            throw new text.Error("Unsupported GB-18030 encoding.");
        }
        if (length > 4 && (length % 2) === 0 && (buffer[0] === 0x00 || buffer[1] === 0x00 || buffer[2] === 0x00 || buffer[3] === 0x00)) {
            const lo = new Uint32Array(256);
            const hi = new Uint32Array(256);
            const size = Math.min(1024, length);
            for (let i = 0; i < size; i += 2) {
                lo[buffer[i]]++;
                hi[buffer[i + 1]]++;
            }
            if (lo[0x00] === 0 && (hi[0x00] / (length >> 1)) > 0.5) {
                assert(encoding, 'utf-16');
                return new text.Decoder.Utf16LE(buffer, 0);
            }
            if (hi[0x00] === 0 && (lo[0x00] / (length >> 1)) > 0.5) {
                assert(encoding, 'utf-16');
                return new text.Decoder.Utf16BE(buffer, 0);
            }
        }
        if (encoding && (encoding.startsWith('iso-8859-') || encoding.startsWith('latin-'))) {
            return new text.Decoder.Latin1(buffer, 0);
        }
        assert(encoding, 'utf-8');
        return new text.Decoder.Utf8(buffer, 0, encoding === 'utf-8');
    }
};

text.Decoder.String = class {

    constructor(buffer) {
        this.buffer = /[\u0020-\uD800]/.test(buffer) ? buffer : buffer.match(/[\uD800-\uDBFF][\uDC00-\uDFFF]|[^\uD800-\uDFFF]/g);
        this.position = 0;
        this.length = this.buffer.length;
    }

    get encoding() {
        return null;
    }

    decode() {
        if (this.position < this.length) {
            return this.buffer[this.position++];
        }
        return undefined;
    }
};

text.Decoder.Utf8 = class {

    constructor(buffer, position, fatal) {
        this.position = position || 0;
        this.buffer = buffer;
        this.fatal = fatal;
    }

    get encoding() {
        return 'utf-8';
    }

    decode() {
        const c = this.buffer[this.position];
        if (c === undefined) {
            return c;
        }
        this.position++;
        if (c < 0x80) {
            return String.fromCodePoint(c);
        }
        if (c >= 0xC2 && c <= 0xDF) {
            if (this.buffer[this.position] !== undefined) {
                const c2 = this.buffer[this.position];
                this.position++;
                return String.fromCharCode(((c & 0x1F) << 6) | (c2 & 0x3F));
            }
        }
        if (c >= 0xE0 && c <= 0xEF) {
            if (this.buffer[this.position + 1] !== undefined) {
                const c2 = this.buffer[this.position];
                if ((c !== 0xE0 || c2 >= 0xA0) && (c !== 0xED || c2 <= 0x9f)) {
                    const c3 = this.buffer[this.position + 1];
                    if (c3 >= 0x80 && c3 < 0xFB) {
                        this.position += 2;
                        return String.fromCharCode(((c & 0x0F) << 12) | ((c2 & 0x3F) << 6) | ((c3 & 0x3F) << 0));
                    }
                }
            }
        }
        if (c >= 0xF0 && c <= 0xF4) {
            if (this.buffer[this.position + 2] !== undefined) {
                const c2 = this.buffer[this.position];
                if (c2 >= 0x80 && c2 <= 0xBF) {
                    const c3 = this.buffer[this.position + 1];
                    if (c3 >= 0x80 && c3 <= 0xBF) {
                        const c4 = this.buffer[this.position + 2];
                        if (c4 >= 0x80 && c4 <= 0xBF) {
                            const codePoint = ((c & 0x07) << 18) | ((c2 & 0x3F) << 12) | ((c3 & 0x3F) << 6) | (c4 & 0x3F);
                            if (codePoint <= 0x10FFFF) {
                                this.position += 3;
                                return String.fromCodePoint(codePoint);
                            }
                        }
                    }
                }
            }
        }
        if (this.fatal) {
            throw new text.Error('Invalid utf-8 character.');
        }
        return String.fromCharCode(0xfffd);
    }
};

text.Decoder.Latin1 = class {

    constructor(buffer, position) {
        this.position = position || 0;
        this.buffer = buffer;
    }

    get encoding() {
        return 'latin-1';
    }

    decode() {
        const c = this.buffer[this.position];
        if (c === undefined) {
            return c;
        }
        this.position++;
        return String.fromCodePoint(c);
    }
};

text.Decoder.Utf16LE = class {

    constructor(buffer, position) {
        this.buffer = buffer;
        this.position = position || 0;
        this.length = buffer.length;
    }

    get encoding() {
        return 'utf-16';
    }

    decode() {
        if (this.position + 1 < this.length) {
            const c = this.buffer[this.position++] | (this.buffer[this.position++] << 8);
            if (c < 0xD800 || c >= 0xDFFF) {
                return String.fromCharCode(c);
            }
            if (c >= 0xD800 && c < 0xDBFF) {
                if (this.position + 1 < this.length) {
                    const c2 = this.buffer[this.position++] | (this.buffer[this.position++] << 8);
                    if (c >= 0xDC00 || c < 0xDFFF) {
                        return String.fromCodePoint(0x10000 + ((c & 0x3ff) << 10) + (c2 & 0x3ff));
                    }
                }
            }
            return String.fromCharCode(0xfffd);
        }
        return undefined;
    }
};

text.Decoder.Utf16BE = class {

    constructor(buffer, position) {
        this.buffer = buffer;
        this.position = position || 0;
        this.length = buffer.length;
    }

    get encoding() {
        return 'utf-16';
    }

    decode() {
        if (this.position + 1 < this.length) {
            const c = (this.buffer[this.position++] << 8) | this.buffer[this.position++];
            if (c < 0xD800 || c >= 0xDFFF) {
                return String.fromCharCode(c);
            }
            if (c >= 0xD800 && c < 0xDBFF) {
                if (this.position + 1 < this.length) {
                    const c2 = (this.buffer[this.position++] << 8) | this.buffer[this.position++];
                    if (c >= 0xDC00 || c < 0xDFFF) {
                        return String.fromCodePoint(0x10000 + ((c & 0x3ff) << 10) + (c2 & 0x3ff));
                    }
                }
            }
            return String.fromCharCode(0xfffd);
        }
        return undefined;
    }
};

text.Decoder.Utf32LE = class {

    constructor(buffer, position) {
        this.buffer = buffer;
        this.position = position || 0;
        this.length = buffer.length;
    }

    get encoding() {
        return 'utf-32';
    }

    decode() {
        if (this.position + 3 < this.length) {
            const c = this.buffer[this.position++] | (this.buffer[this.position++] << 8) || (this.buffer[this.position++] << 16) || (this.buffer[this.position++] << 24);
            if (c < 0x10FFFF) {
                return String.fromCodePoint(c);
            }
            return String.fromCharCode(0xfffd);
        }
        return undefined;
    }
};

text.Decoder.Utf32BE = class {

    constructor(buffer, position) {
        this.buffer = buffer;
        this.position = position || 0;
        this.length = buffer.length;
    }

    get encoding() {
        return 'utf-32';
    }

    decode() {
        if (this.position + 3 < this.length) {
            const c = (this.buffer[this.position++] << 24) || (this.buffer[this.position++] << 16) || (this.buffer[this.position++] << 8) | this.buffer[this.position++];
            if (c < 0x10FFFF) {
                return String.fromCodePoint(c);
            }
            return String.fromCharCode(0xfffd);
        }
        return undefined;
    }
};

text.Reader = class {

    constructor(data) {
        this.decoder = text.Decoder.open(data);
        this.position = 0;
        this.length = Number.MAX_SAFE_INTEGER;
    }

    static open(data, length) {
        return new text.Reader(data, length);
    }

    read(terminal) {
        if (this.position >= this.length) {
            return undefined;
        }
        let line = '';
        let buffer = null;
        for (;;) {
            const c = this.decoder.decode();
            if (c === undefined) {
                this.length = this.position;
                break;
            }
            this.position++;
            if (c === terminal || this.position > this.length) {
                break;
            }
            line += c;
            if (line.length >= 64) {
                buffer = buffer || [];
                buffer.push(line);
                line = '';
            }
        }
        if (buffer) {
            buffer.push(line);
            return buffer.join('');
        }
        return line;
    }
};

text.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Text Error';
    }
};

export const Decoder = text.Decoder;
export const Reader = text.Reader;
