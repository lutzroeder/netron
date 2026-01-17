
const zip = {};
const gzip = {};
const zlib = {};

zip.Archive = class {

    static async import() {
        if (typeof process === 'object' && typeof process.versions === 'object' && typeof process.versions.node !== 'undefined') {
            zip.zlib = await import('zlib');
        }
    }

    static open(data, format) {
        const stream = data instanceof Uint8Array ? new zip.BinaryReader(data) : data;
        if (stream && stream.length > 2) {
            const buffer = stream.peek(Math.min(512, stream.length));
            if (buffer.length >= 512) {
                // Reject tar with ZIP content
                const sum = buffer.map((value, index) => (index >= 148 && index < 156) ? 32 : value).reduce((a, b) => a + b, 0);
                const checksum = parseInt(Array.from(buffer.slice(148, 156)).map((c) => String.fromCharCode(c)).join('').split('\0').shift(), 8);
                if (!isNaN(checksum) && sum === checksum) {
                    return null;
                }
            }
            if ((!format || format === 'zlib') && buffer[0] === 0x78) { // zlib
                const check = (buffer[0] << 8) + buffer[1];
                if (check % 31 === 0) {
                    return new zlib.Archive(stream);
                }
            }
            if ((!format || format === 'gzip') && buffer.length >= 18 && buffer[0] === 0x1f && buffer[1] === 0x8b) { // gzip
                return new gzip.Archive(stream);
            }
            if (!format || format === 'zip') {
                const search = buffer[0] === 0x50 && buffer[1] === 0x4B;
                const location = stream.position;
                const seek = (signature, size) => {
                    signature = Array.from(signature, (c) => c.charCodeAt(0));
                    let position = stream.length;
                    const offset = Math.max(stream.length - 0x1000000, 0);
                    while (position > offset) {
                        position = Math.max(0, position - 66000);
                        stream.seek(position);
                        const length = Math.min(stream.length - position, 66000 + size);
                        const buffer = stream.read(length);
                        for (let i = length - size; i >= 0; i--) {
                            i = buffer.lastIndexOf(signature[0], i);
                            if (i !== -1 && signature[1] === buffer[i + 1] && signature[2] === buffer[i + 2] && signature[3] === buffer[i + 3]) {
                                stream.seek(position + i);
                                return new zip.BinaryReader(buffer.subarray(i, i + size));
                            }
                        }
                        if (!search) {
                            break;
                        }
                    }
                    return null;
                };
                const read = (signature, size) => {
                    if ((stream.position - size) >= 0) {
                        stream.skip(-size);
                        signature = Array.from(signature, (c) => c.charCodeAt(0));
                        const buffer = stream.peek(size);
                        if (buffer[0] === signature[0] && buffer[1] === signature[1] && buffer[2] === signature[2] && buffer[3] === signature[3]) {
                            return new zip.BinaryReader(buffer);
                        }
                    }
                    return null;
                };
                const header = {};
                let position = -1;
                let reader = seek('PK\x05\x06', 22);
                if (reader) {
                    position = stream.position;
                    reader.skip(4);
                    header.disk = reader.uint16();
                    header.startDisk = reader.uint16();
                    header.diskRecords = reader.uint16();
                    header.totalRecords = reader.uint16();
                    header.size = reader.uint32();
                    header.offset = reader.uint32();
                    header.commentLength = reader.uint16();
                    reader = null;
                    if (read('PK\x06\x07', 20)) {
                        reader = read('PK\x06\x06', 56);
                        if (!reader) {
                            stream.seek(location);
                            throw new zip.Error('Invalid ZIP data. ZIP64 end of central directory not found.');
                        }
                    }
                } else {
                    reader = seek('PK\x06\x06', 56);
                    if (!reader) {
                        stream.seek(location);
                        if (search) {
                            throw new zip.Error('Invalid ZIP data. End of central directory not found.');
                        }
                        return null;
                    }
                }
                if (reader) {
                    position = stream.position;
                    reader.skip(4);
                    reader.recordSize = reader.uint64();
                    reader.version = reader.uint16();
                    reader.minVersion = reader.uint16();
                    reader.disks = reader.uint32();
                    reader.startDisk = reader.uint32();
                    header.diskRecords = reader.uint64();
                    header.totalRecords = reader.uint64();
                    header.size = reader.uint64().toNumber();
                    header.offset = reader.uint64();
                    if (header.offset > Number.MAX_SAFE_INTEGER) {
                        stream.seek(location);
                        throw new zip.Error('ZIP 64-bit central directory offset not supported.');
                    }
                    header.offset = header.offset.toNumber();
                }
                position -= header.size;
                if (position < 0 || position > stream.length) {
                    stream.seek(location);
                    throw new zip.Error('Invalid ZIP data. Central directory size is outside expected range.');
                }
                if (position < header.offset) {
                    stream.seek(location);
                    throw new zip.Error('Invalid ZIP data. Central directory offset is outside expected range.');
                }
                stream.seek(position);
                position -= header.offset;
                const archive = new zip.Archive(stream, position);
                stream.seek(location);
                return archive;
            }
        }
        return null;
    }

    constructor(stream, offset) {
        offset = offset || 0;
        this._entries = new Map();
        const headers = [];
        const signature = Array.from('PK\x01\x02', (c) => c.charCodeAt(0));
        while (stream.position + 4 < stream.length && stream.read(4).every((value, index) => value === signature[index])) {
            const header = {};
            const reader = new zip.BinaryReader(stream.read(42));
            reader.uint16(); // version made by
            reader.skip(2); // version needed to extract
            const flags = reader.uint16();
            if ((flags & 1) === 1) {
                throw new zip.Error('Encrypted ZIP entries not supported.');
            }
            header.encoding = flags & 0x800 ? 'utf-8' : 'ascii';
            header.compressionMethod = reader.uint16();
            header.date = reader.uint32(); // date
            header.crc32 = reader.uint32(); // crc32
            header.compressedSize = reader.uint32();
            header.size = reader.uint32();
            header.nameLength = reader.uint16(); // file name length
            const extraDataLength = reader.uint16();
            const commentLength = reader.uint16();
            header.disk = reader.uint16(); // disk number start
            reader.uint16(); // internal file attributes
            reader.uint32(); // external file attributes
            header.localHeaderOffset = reader.uint32();
            const nameBuffer = stream.read(header.nameLength);
            const decoder = new TextDecoder(header.encoding);
            header.name = decoder.decode(nameBuffer);
            const extraData = stream.read(extraDataLength);
            if (extraData.length > 0) {
                const reader = new zip.BinaryReader(extraData);
                while (reader.position < reader.length) {
                    const type = reader.uint16();
                    const length = reader.uint16();
                    switch (type) {
                        case 0x0001:
                            if (header.size === 0xffffffff) {
                                header.size = reader.uint64().toNumber();
                                if (header.size === undefined) {
                                    throw new zip.Error('ZIP 64-bit size not supported.');
                                }
                            }
                            if (header.compressedSize === 0xffffffff) {
                                header.compressedSize = reader.uint64().toNumber();
                                if (header.compressedSize === undefined) {
                                    throw new zip.Error('ZIP 64-bit compressed size not supported.');
                                }
                            }
                            if (header.localHeaderOffset === 0xffffffff) {
                                header.localHeaderOffset = reader.uint64().toNumber();
                                if (header.localHeaderOffset === undefined) {
                                    throw new zip.Error('ZIP 64-bit offset not supported.');
                                }
                            }
                            if (header.disk === 0xffff) {
                                header.disk = reader.uint32();
                            }
                            break;
                        default:
                            reader.skip(length);
                            break;
                    }
                }
            }
            stream.read(commentLength); // comment
            headers.push(header);
        }
        for (const header of headers) {
            if (header.size === 0 && header.name.endsWith('/')) {
                continue;
            }
            const entry = new zip.Entry(stream, header, offset);
            this._entries.set(entry.name, entry.stream);
        }
    }

    get entries() {
        return this._entries;
    }
};

zip.Entry = class {

    constructor(stream, header, offset) {
        offset = offset || 0;
        this._name = header.name;
        stream.seek(offset + header.localHeaderOffset);
        if (stream.position + 4 > stream.length || String.fromCharCode(...stream.read(4)) !== 'PK\x03\x04') {
            this._stream = new zip.ErrorStream(header.size, 'Invalid ZIP data. Local file header signature not found.');
            return;
        }
        const reader = new zip.BinaryReader(stream.read(26));
        reader.skip(22);
        header.nameLength = reader.uint16();
        const extraDataLength = reader.uint16();
        header.nameBuffer = stream.read(header.nameLength);
        stream.skip(extraDataLength);
        const decoder = new TextDecoder(header.encoding);
        this._name = decoder.decode(header.nameBuffer);
        this._stream = stream.stream(header.compressedSize);
        switch (header.compressionMethod) {
            case 0: // stored
                if (header.size !== header.compressedSize) {
                    this._stream = new zip.ErrorStream(header.size, 'Invalid ZIP entry compression size.');
                }
                break;
            case 8: // deflate
                this._stream = new zip.InflaterStream(this._stream, header.size);
                break;
            default:
                this._stream = new zip.ErrorStream(header.size, `Invalid ZIP entry compression method '${header.compressionMethod}'.`);
                break;
        }
    }

    get name() {
        return this._name;
    }

    get stream() {
        return this._stream;
    }
};

zip.Inflater = class {

    inflateRaw(data, length, size) {
        let buffer = null;
        if (zip.zlib && size === undefined && (length === undefined || length > 0x4000)) {
            buffer = zip.zlib.inflateRawSync(data);
        } else {
            const reader = new zip.BitReader(data);
            const writer = length === undefined || size !== undefined ? new zip.BlockWriter() : new zip.BufferWriter(length);
            if (!zip.Inflater._staticLengthTree) {
                zip.Inflater._codeLengths = new Uint8Array(19);
                zip.Inflater._codeOrder = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                zip.Inflater._lengthBase = [24, 32, 40, 48, 56, 64, 72, 80, 89, 105, 121, 137, 154,  186,  218,  250,  283,  347, 411,  475,  540,  668,  796,  924, 1053, 1309, 1565, 1821, 2064, 7992, 7992, 7992];
                zip.Inflater._distanceBase = [16, 32, 48, 64, 81, 113, 146, 210, 275, 403, 532, 788, 1045, 1557, 2070, 3094, 4119, 6167, 8216, 12312, 16409, 24601, 32794, 49178, 65563, 98331, 131100, 196636, 262173, 393245, 1048560, 1048560];
            }
            let type = 0;
            do {
                type = reader.bits(3);
                switch (type >>> 1) {
                    case 0: { // uncompressed block
                        this._copyUncompressedBlock(reader, writer);
                        break;
                    }
                    case 1: { // block with fixed huffman trees
                        if (!zip.Inflater._staticLengthTree) {
                            zip.Inflater._staticLengthTree = zip.HuffmanTree.create(new Uint8Array([].concat(...[[144, 8], [112, 9], [24, 7], [8, 8]].map((x) => [...Array(x[0])].map(() => x[1])))));
                            zip.Inflater._staticDistanceTree = zip.HuffmanTree.create(new Uint8Array([...Array(32)].map(() => 5)));
                        }
                        this._lengthTree = zip.Inflater._staticLengthTree;
                        this._distanceTree = zip.Inflater._staticDistanceTree;
                        this._inflateBlock(reader, writer);
                        break;
                    }
                    case 2: { // block with dynamic huffman trees
                        this._decodeTrees(reader);
                        this._inflateBlock(reader, writer);
                        break;
                    }
                    default: {
                        throw new zip.Error('Unsupported block type.');
                    }
                }
                if (size !== undefined && writer.length >= size) {
                    break;
                }
            } while ((type & 1) === 0);
            if (size === undefined && length !== undefined && length !== writer.length) {
                throw new zip.Error('Invalid uncompressed size.');
            }
            buffer = writer.toBuffer();
        }
        if (size === undefined && length !== undefined && length !== buffer.length) {
            throw new zip.Error('Invalid uncompressed size.');
        }
        return buffer;
    }

    _copyUncompressedBlock(reader, writer) {
        const length = reader.uint16();
        const inverseLength = reader.uint16();
        if (length !== (~inverseLength & 0xffff)) {
            throw new zip.Error('Invalid uncompressed block length.');
        }
        writer.write(reader.read(length));
    }

    _decodeTrees(reader) {
        const hlit = reader.bits(5) + 257;
        const hdist = reader.bits(5) + 1;
        const hclen = reader.bits(4) + 4;
        const codeLengths = zip.Inflater._codeLengths;
        for (let i = 0; i < codeLengths.length; i++) {
            codeLengths[i] = 0;
        }
        const codeOrder = zip.Inflater._codeOrder;
        for (let i = 0; i < hclen; i++) {
            codeLengths[codeOrder[i]] = reader.bits(3);
        }
        const codeTree = zip.HuffmanTree.create(codeLengths);
        const codeMask = codeTree.length - 1;
        const lengths = new Uint8Array(hlit + hdist);
        let value = 0;
        let length = 0;
        for (let i = 0; i < hlit + hdist;) {
            const code = codeTree[reader.bits16() & codeMask];
            reader.position += code & 0x0f;
            const literal = code >>> 4;
            switch (literal) {
                case 16: length = reader.bits(2) + 3; break;
                case 17: length = reader.bits(3) + 3; value = 0; break;
                case 18: length = reader.bits(7) + 11; value = 0; break;
                default: length = 1; value = literal; break;
            }
            for (; length > 0; length--) {
                lengths[i++] = value;
            }
        }
        this._lengthTree = zip.HuffmanTree.create(lengths.subarray(0, hlit));
        this._distanceTree = zip.HuffmanTree.create(lengths.subarray(hlit, hlit + hdist));
    }

    _inflateBlock(reader, writer) {
        const lengthTree = this._lengthTree;
        const distanceTree = this._distanceTree;
        const lengthMask = lengthTree.length - 1;
        const distanceMask = distanceTree.length - 1;
        const buffer = writer.buffer;
        const threshold = writer.threshold === undefined ? writer.length : writer.threshold;
        let position = writer.position;
        for (;;) {
            if (position > threshold) {
                position = writer.push(position);
            }
            const code = lengthTree[reader.bits16() & lengthMask];
            reader.position += code & 0x0f;
            const literal = code >>> 4;
            if (literal < 256) {
                buffer[position++] = literal;
            } else if (literal === 256) {
                writer.push(position);
                return;
            } else {
                let length = literal - 254;
                if (literal > 264) {
                    const lengthBase = zip.Inflater._lengthBase[literal - 257];
                    length = (lengthBase >>> 3) + reader.bits(lengthBase & 0x07);
                }
                const code = distanceTree[reader.bits16() & distanceMask];
                reader.position += code & 0x0f;
                const distanceBase = zip.Inflater._distanceBase[code >>> 4];
                const bits = distanceBase & 0x0f;
                const distance = (distanceBase >>> 4) + (reader.bits16() & ((1 << bits) - 1));
                reader.position += bits;
                let offset = position - distance;
                for (let i = 0; i < length; i++) {
                    buffer[position++] = buffer[offset++];
                }
            }
        }
    }
};

zip.HuffmanTree = class {

    static create(tree) {
        const bits = Math.max.apply(null, tree);
        // Algorithm from https://github.com/photopea/UZIP.js
        let rev15 = zip.HuffmanTree._rev15;
        if (!rev15) {
            const length = 1 << 15;
            rev15 = new Uint16Array(length);
            for (let i = 0; i < length; i++) {
                let x = i;
                x = (((x & 0xaaaaaaaa) >>> 1) | ((x & 0x55555555) << 1));
                x = (((x & 0xcccccccc) >>> 2) | ((x & 0x33333333) << 2));
                x = (((x & 0xf0f0f0f0) >>> 4) | ((x & 0x0f0f0f0f) << 4));
                x = (((x & 0xff00ff00) >>> 8) | ((x & 0x00ff00ff) << 8));
                rev15[i] = (((x >>> 16) | (x << 16))) >>> 17;
            }
            zip.HuffmanTree._rev15 = rev15;
            zip.HuffmanTree._bitLengthCounts = new Uint16Array(16);
            zip.HuffmanTree._nextCodes = new Uint16Array(16);
        }
        const length = tree.length;
        const bitLengthCounts = zip.HuffmanTree._bitLengthCounts;
        for (let i = 0; i < 16; i++) {
            bitLengthCounts[i] = 0;
        }
        for (let i = 0; i < length; i++) {
            bitLengthCounts[tree[i]]++;
        }
        const nextCodes = zip.HuffmanTree._nextCodes;
        let code = 0;
        bitLengthCounts[0] = 0;
        for (let i = 0; i < bits; i++) {
            code = (code + bitLengthCounts[i]) << 1;
            nextCodes[i + 1] = code;
        }
        const codes = new Uint16Array(length);
        for (let i = 0; i < length; i++) {
            const index = tree[i];
            if (index !== 0) {
                codes[i] = nextCodes[index];
                nextCodes[index]++;
            }
        }
        const shift = 15 - bits;
        const table = new Uint16Array(1 << bits);
        for (let i = 0; i < length; i++) {
            const c = tree[i];
            if (c !== 0) {
                const value = (i << 4) | c;
                const rest = bits - c;
                let index = codes[i] << rest;
                const max = index + (1 << rest);
                for (; index !== max; index++) {
                    table[rev15[index] >>> shift] = value;
                }
            }
        }
        return table;
    }
};

zip.BitReader = class {

    constructor(buffer) {
        this.buffer = buffer;
        this.position = 0;
    }

    bits(count) {
        const offset = Math.floor(this.position / 8);
        const shift = (this.position & 7);
        this.position += count;
        return ((this.buffer[offset] | (this.buffer[offset + 1] << 8)) >>> shift) & ((1 << count) - 1);
    }

    bits16() {
        const offset = Math.floor(this.position / 8);
        return ((this.buffer[offset] | (this.buffer[offset + 1] << 8) | (this.buffer[offset + 2] << 16)) >>> (this.position & 7));
    }

    read(length) {
        const remainder = this.position & 7;
        if (remainder !== 0) {
            this.position += (8 - remainder);
        }
        const offset = Math.floor(this.position / 8);
        this.position += length * 8;
        return this.buffer.subarray(offset, offset + length);
    }

    uint16() {
        const remainder = this.position & 7;
        if (remainder !== 0) {
            this.position += (8 - remainder);
        }
        const offset = Math.floor(this.position / 8);
        this.position += 16;
        return this.buffer[offset] | (this.buffer[offset + 1] << 8);
    }
};

zip.BlockWriter = class {

    constructor() {
        this.blocks = [];
        this.buffer = new Uint8Array(65536);
        this.position = 0;
        this.length = 0;
        this.threshold = 0xf400;
    }

    push(position) {
        this.blocks.push(new Uint8Array(this.buffer.subarray(this.position, position)));
        this.length += position - this.position;
        this.position = position;
        return this._reset();
    }

    write(buffer) {
        this.blocks.push(buffer);
        const length = buffer.length;
        this.length += length;
        if (length > 32768) {
            this.buffer.set(buffer.subarray(length - 32768, length), 0);
            this.position = 32768;
        } else {
            this._reset();
            this.buffer.set(buffer, this.position);
            this.position += length;
        }
    }

    toBuffer() {
        const buffer = new Uint8Array(this.length);
        let offset = 0;
        for (const block of this.blocks) {
            buffer.set(block, offset);
            offset += block.length;
        }
        return buffer;
    }

    _reset() {
        if (this.position > 32768) {
            this.buffer.set(this.buffer.subarray(this.position - 32768, this.position), 0);
            this.position = 32768;
        }
        return this.position;
    }
};

zip.BufferWriter = class {

    constructor(length) {
        this.buffer = new Uint8Array(length);
        this.length = length;
        this.position = 0;
    }

    push(position) {
        this.position = position;
        if (this.position > this.length) {
            throw new zip.Error('Invalid size.');
        }
        return this.position;
    }

    write(buffer) {
        this.buffer.set(buffer, this.position);
        this.position += buffer.length;
        if (this.position > this.length) {
            throw new zip.Error('Invalid size.');
        }
        return this.position;
    }

    toBuffer() {
        return this.buffer;
    }
};

zip.InflaterStream = class {

    constructor(stream, length) {
        this._stream = stream;
        this._offset = this._stream.position;
        this._position = 0;
        this._length = length;
    }

    get position() {
        return this._position;
    }

    get length() {
        if (this._length === undefined) {
            this._inflate();
        }
        return this._length;
    }

    seek(position) {
        if (position !== this._position) {
            this._inflate(position, 0);
            this._position = position >= 0 ? position : this._length + position;
        }
    }

    skip(offset) {
        this._inflate(this.position, offset);
        this._position += offset;
    }

    peek(length) {
        const position = this._position;
        length = length === undefined ? this.length - position : length;
        this.skip(length);
        const end = this._position;
        this.seek(position);
        if (position === 0 && length === this.length) {
            return this._buffer;
        }
        return this._buffer.subarray(position, end);
    }

    read(length) {
        const position = this._position;
        length = length === undefined ? this.length - position : length;
        this.skip(length);
        if (position === 0 && length === this.length) {
            return this._buffer;
        }
        return this._buffer.subarray(position, this._position);
    }

    stream(length) {
        const buffer = this.read(length);
        return new zip.BinaryReader(buffer);
    }

    _inflate(position, length) {
        const size = Number.isInteger(position) && Number.isInteger(length) ? position + length : undefined;
        if (this._buffer === undefined || (size !== undefined && this._buffer.length < size)) {
            const position = this._stream.position;
            this._stream.seek(this._offset);
            const buffer = this._stream.peek();
            this._buffer = new zip.Inflater().inflateRaw(buffer, this._length, size);
            this._stream.seek(position);
            if ((size === undefined || this._buffer.length > size) && (this._length === undefined)) {
                this._length = this._buffer.length;
                delete this._stream;
            }
        }
    }
};

zip.ErrorStream = class {

    constructor(size, message) {
        this._message = message;
        this._position = 0;
        this._length = size;
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
        if (this._position > this._length || this._position < 0) {
            throw new zip.Error('Invalid ZIP data. Unexpected end of file.');
        }
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._length || this._position < 0) {
            throw new zip.Error('Invalid ZIP data. Unexpected end of file.');
        }
    }

    peek(/* length */) {
        this._throw();
    }

    read(/* length */) {
        this._throw();
    }

    stream(/* length */) {
        this._throw();
    }

    _throw() {
        throw new zip.Error(this._message);
    }
};

zip.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._length = buffer.length;
        this._position = 0;
        this._view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    create(buffer) {
        return new zip.BinaryReader(buffer);
    }

    stream(length) {
        return this.create(this.read(length));
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
    }

    skip(offset) {
        this._position += offset;
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

    byte() {
        const position = this._position;
        this.skip(1);
        return this._buffer[position];
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
        return this._view.getBigUint64(position, true);
    }
};

zlib.Archive = class {

    constructor(stream) {
        const position = stream.position;
        stream.read(2);
        this._entries = new Map([['', new zip.InflaterStream(stream)]]);
        stream.seek(position);
    }

    get entries() {
        return this._entries;
    }
};

zip.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'ZIP Error';
    }
};

gzip.Archive = class {

    constructor(stream) {
        const position = stream.position;
        if (stream.position + 10 > stream.length) {
            throw new gzip.Error('Invalid gzip header size.');
        }
        const header = stream.peek(10);
        if (header[0] !== 0x1f || header[1] !== 0x8b) {
            throw new gzip.Error('Invalid gzip signature.');
        }
        if (header[2] !== 8) {
            stream.seek(position);
            throw new gzip.Error(`Invalid compression method '${header[2]}'.`);
        }
        stream.skip(10);
        const string = () => {
            let content = '';
            while (stream.position < stream.length) {
                const [value] = stream.read(1);
                if (value === 0x00) {
                    break;
                }
                content += String.fromCharCode(value);
            }
            return content;
        };
        const fhcrc = header[3] & 2;
        const fextra = header[3] & 4;
        const fname = header[3] & 8;
        const fcomment = header[3] & 16;
        if (fextra) {
            const buffer = stream.read(2);
            const xlen = buffer[0] | (buffer[1] << 8);
            stream.skip(xlen);
        }
        const name = fname ? string() : '';
        if (fcomment) {
            string();
        }
        if (fhcrc) {
            stream.skip(2);
        }
        this._entries = new Map();
        this._entries.set(name, new gzip.InflaterStream(stream));
        stream.seek(position);
    }

    get entries() {
        return this._entries;
    }
};

gzip.InflaterStream = class {

    constructor(stream) {
        this._stream = stream.stream(stream.length - stream.position - 8);
        const reader = new zip.BinaryReader(stream.read(8));
        reader.uint32(); // CRC32
        this._length = reader.uint32(); // ISIZE
        this._position = 0;
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    seek(position) {
        if (this._buffer === undefined) {
            this._inflate();
        }
        this._position = position >= 0 ? position : this._length + position;
    }

    skip(offset) {
        if (this._buffer === undefined) {
            this._inflate();
        }
        this._position += offset;
    }

    stream(length) {
        return new zip.BinaryReader(this.read(length));
    }

    peek(length) {
        const position = this._position;
        length = length === undefined ? this._length - this._position : length;
        this.skip(length);
        const end = this._position;
        this.seek(position);
        if (position === 0 && length === this._length) {
            return this._buffer;
        }
        return this._buffer.subarray(position, end);
    }

    read(length) {
        const position = this._position;
        length = length === undefined ? this._length - this._position : length;
        this.skip(length);
        if (position === 0 && length === this._length) {
            return this._buffer;
        }
        return this._buffer.subarray(position, this._position);
    }

    _inflate() {
        if (this._buffer === undefined) {
            const buffer = this._stream.peek();
            this._buffer = new zip.Inflater().inflateRaw(buffer, this._length);
            delete this._stream;
        }
    }
};

gzip.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'gzip Error';
    }
};

export const Archive = zip.Archive;
