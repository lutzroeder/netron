/* jshint esversion: 6 */

var zip = zip || {};

zip.Archive = class {

    constructor(buffer) {
        this._entries = [];
        const stream = buffer instanceof Uint8Array ? new zip.BinaryReader(buffer) : buffer;
        const signature = [ 0x50, 0x4B, 0x01, 0x02 ];
        if (stream.length < 4 || !stream.peek(2).every((value, index) => value === signature[index])) {
            throw new zip.Error('Invalid Zip archive.');
        }
        const lookup = (stream, signature) => {
            let position = stream.length - 65536;
            while (position !== 0) {
                position = Math.max(0, position);
                stream.seek(position);
                const buffer = stream.read(Math.min(stream.length - position, 65000));
                for (let i = buffer.length - 4; i >= 0; i--) {
                    if (signature[0] === buffer[i] &&
                        signature[1] === buffer[i + 1] &&
                        signature[2] === buffer[i + 2] &&
                        signature[3] === buffer[i + 3]) {
                        stream.seek(position + i + 4);
                        return true;
                    }
                }
                position += 4000;
            }
            return false;
        };
        if (!lookup(stream, [ 0x50, 0x4B, 0x05, 0x06 ])) {
            throw new zip.Error('End of central directory not found.');
        }
        let reader = new zip.BinaryReader(stream.read(16));
        reader.skip(12);
        let offset = reader.uint32(); // central directory offset
        if (offset > stream.length) {
            if (!lookup(stream, [ 0x50, 0x4B, 0x06, 0x06 ])) {
                throw new zip.Error('Zip64 end of central directory not found.');
            }
            reader = new zip.BinaryReader(stream.read(52));
            reader.skip(44);
            offset = reader.uint32();
            if (reader.uint32() !== 0) {
                throw new zip.Error('Zip 64-bit central directory offset not supported.');
            }
        }
        if (offset > stream.length) {
            throw new zip.Error('Invalid central directory offset.');
        }
        stream.seek(offset); // central directory offset

        const entries = [];
        while (stream.position + 4 < stream.length && stream.read(4).every((value, index) => value === signature[index])) {
            const entry = {};
            const reader = new zip.BinaryReader(stream.read(42));
            reader.uint16(); // version made by
            reader.skip(2); // version needed to extract
            const flags = reader.uint16();
            if ((flags & 1) == 1) {
                throw new zip.Error('Encrypted entries not supported.');
            }
            entry.compressionMethod = reader.uint16();
            reader.uint32(); // date
            reader.uint32(); // crc32
            entry.compressedSize = reader.uint32();
            entry.size = reader.uint32();
            entry.nameLength = reader.uint16(); // file name length
            const extraDataLength = reader.uint16();
            const commentLength = reader.uint16();
            entry.disk = reader.uint16(); // disk number start
            reader.uint16(); // internal file attributes
            reader.uint32(); // external file attributes
            entry.localHeaderOffset = reader.uint32();
            entry.nameBuffer = stream.read(entry.nameLength);
            const extraData = stream.read(extraDataLength);
            if (extraData.length > 0) {
                const reader = new zip.BinaryReader(extraData);
                while (reader.position < reader.length) {
                    const type = reader.uint16();
                    reader.uint16(); // length
                    switch (type) {
                        case 0x0001:
                            if (entry.size === 0xffffffff) {
                                entry.size = reader.uint32();
                                if (reader.uint32() !== 0) {
                                    throw new zip.Error('Zip 64-bit offset not supported.');
                                }
                            }
                            if (entry.compressedSize === 0xffffffff) {
                                entry.compressedSize = reader.uint32();
                                if (reader.uint32() !== 0) {
                                    throw new zip.Error('Zip 64-bit offset not supported.');
                                }
                            }
                            if (entry.localHeaderOffset === 0xffffffff) {
                                entry.localHeaderOffset = reader.uint32();
                                if (reader.uint32() !== 0) {
                                    throw new zip.Error('Zip 64-bit offset not supported.');
                                }
                            }
                            if (entry.disk === 0xffff) {
                                entry.disk = reader.uint32();
                            }
                            break;
                    }
                }
            }
            stream.read(commentLength); // comment
            entries.push(entry);
        }
        for (const entry of entries) {
            this._entries.push(new zip.Entry(stream, entry));
        }
        stream.seek(0);
    }

    get entries() {
        return this._entries;
    }
};

zip.Entry = class {

    constructor(stream, entry) {
        stream.seek(entry.localHeaderOffset);
        const signature = [ 0x50, 0x4B, 0x03, 0x04 ];
        if (stream.position + 4 > stream.length || !stream.read(4).every((value, index) => value === signature[index])) {
            throw new zip.Error('Invalid local file header signature.');
        }
        const reader = new zip.BinaryReader(stream.read(26));
        reader.skip(22);
        entry.nameLength = reader.uint16();
        const extraDataLength = reader.uint16();
        entry.nameBuffer = stream.read(entry.nameLength);
        stream.skip(extraDataLength);
        this._name = '';
        for (const c of entry.nameBuffer) {
            this._name += String.fromCharCode(c);
        }
        this._stream = stream.stream(entry.compressedSize);
        switch (entry.compressionMethod) {
            case 0: { // Stored
                if (entry.size !== entry.compressedSize) {
                    throw new zip.Error('Invalid compression size.');
                }
                break;
            }
            case 8: {
                // Deflate
                this._stream = new zip.InflaterStream(this._stream, entry.size);
                break;
            }
            default:
                throw new zip.Error('Invalid compression method.');
        }
    }

    get name() {
        return this._name;
    }

    get stream() {
        return this._stream;
    }

    get data() {
        return this.stream.peek();
    }
};

zip.Inflater = class {

    inflateRaw(input, output) {
        // if (typeof process === 'object' && typeof process.versions == 'object' && typeof process.versions.node !== 'undefined') {
        //     return require('zlib').inflateRawSync(input);
        // }
        zip.Inflater._initialize();
        const fixedLengthExtraBits = zip.Inflater._fixedLengthExtraBits;
        const fixedDistanceExtraBits = zip.Inflater._fixedDistanceExtraBits;
        const lengthBase = zip.Inflater._lengthBase;
        const distanceBase = zip.Inflater._distanceBase;
        const codeLengthIndexMap = zip.Inflater._codeLengthIndexMap;
        const fixedLengthMap = zip.Inflater._fixedLengthMap;
        const fixedDistanceMap = zip.Inflater._fixedDistanceMap;
        const max = function (array) {
            let value = array[0];
            for (let i = 1; i < array.length; ++i) {
                if (array[i] > value) {
                    value = array[i];
                }
            }
            return value;
        };
        const bits = (buffer, position, mask) => {
            const offset = (position / 8) >> 0;
            return ((buffer[offset] | (buffer[offset + 1] << 8)) >>> (position & 7)) & mask;
        };
        const bits16 = (buffer, position) => {
            const offset = (position / 8) >> 0;
            return ((buffer[offset] | (buffer[offset + 1] << 8) | (buffer[offset + 2] << 16)) >>> (position & 7));
        };
        const inputLength = input.length;
        const allocate = !output;
        if (!output) {
            output = new Uint8Array(inputLength * 3);
        }
        const resize = function (length) {
            if (length > output.length) {
                const buffer = new Uint8Array(Math.max(output.length << 1, length));
                buffer.set(output);
                output = buffer;
            }
        };
        let final = 0;
        let position = 0;
        let offset = 0;
        let lengthMap = null;
        let distanceMap = null;
        let maxLengthBits = null;
        let maxDistanceBits = null;
        if (final && !lengthMap) {
            return output;
        }
        const inputBitLength = inputLength * 8;
        do {
            if (!lengthMap) {
                final = bits(input, position, 1);
                const type = bits(input, position + 1, 3);
                position += 3;
                if (type === 0) { // no compression
                    const start = ((position / 8) >> 0) + (position & 7 && 1) + 4;
                    const length = input[start - 4] | (input[start - 3] << 8);
                    const end = start + length;
                    if (end > inputLength) {
                        throw new zip.Error('Unexpected end of file.');
                    }
                    if (allocate) {
                        resize(offset + length);
                    }
                    output.set(input.subarray(start, end), offset);
                    offset += length;
                    position = end * 8;
                    continue;
                }
                else if (type === 1) { // fixed huffman
                    lengthMap = fixedLengthMap;
                    distanceMap = fixedDistanceMap;
                    maxLengthBits = 9;
                    maxDistanceBits = 5;
                }
                else if (type === 2) { // dynamic huffman
                    const literal = bits(input, position, 31) + 257;
                    const lengths = bits(input, position + 10, 15) + 4;
                    const length = literal + bits(input, position + 5, 31) + 1;
                    position += 14;
                    const lengthDistanceTree = new Uint8Array(length);
                    const codeLengthTree = new Uint8Array(19);
                    for (let i = 0; i < lengths; ++i) {
                        codeLengthTree[codeLengthIndexMap[i]] = bits(input, position + i * 3, 7);
                    }
                    position += lengths * 3;
                    const codeLengthBits = max(codeLengthTree);
                    const codeLengthBitsMask = (1 << codeLengthBits) - 1;
                    const codeLengthsMap = zip.Inflater._huffman(codeLengthTree, codeLengthBits);
                    for (let i = 0; i < length;) {
                        const code = codeLengthsMap[bits(input, position, codeLengthBitsMask)];
                        position += code & 15;
                        const symbol = code >>> 4;
                        if (symbol < 16) {
                            lengthDistanceTree[i++] = symbol;
                        }
                        else {
                            let value = 0;
                            let length = 0;
                            if (symbol == 16) {
                                length = 3 + bits(input, position, 3);
                                position += 2;
                                value = lengthDistanceTree[i - 1];
                            }
                            else if (symbol == 17) {
                                length = 3 + bits(input, position, 7);
                                position += 3;
                            }
                            else if (symbol == 18) {
                                length = 11 + bits(input, position, 127);
                                position += 7;
                            }
                            while (length--) {
                                lengthDistanceTree[i++] = value;
                            }
                        }
                    }
                    const lengthTree = lengthDistanceTree.subarray(0, literal);
                    const distanceTree = lengthDistanceTree.subarray(literal);
                    maxLengthBits = max(lengthTree);
                    maxDistanceBits = max(distanceTree);
                    lengthMap = zip.Inflater._huffman(lengthTree, maxLengthBits);
                    distanceMap = zip.Inflater._huffman(distanceTree, maxDistanceBits);
                }
                else {
                    throw new zip.Error('Invalid block type.');
                }
                if (position > inputBitLength) {
                    throw new zip.Error('Unexpected end of file.');
                }
            }
            if (allocate) {
                resize(offset + 131072);
            }
            const maxLengthBitsMask = (1 << maxLengthBits) - 1;
            const maxDistanceBitsMask = (1 << maxDistanceBits) - 1;
            for (;;) {
                const code = lengthMap[bits16(input, position) & maxLengthBitsMask];
                const symbol = code >>> 4;
                position += code & 15;
                if (position > inputBitLength) {
                    throw new zip.Error('Unexpected end of file.');
                }
                if (!code) {
                    throw new zip.Error('Invalid length/literal.');
                }
                if (symbol < 256) {
                    output[offset++] = symbol;
                }
                else if (symbol == 256) {
                    lengthMap = null;
                    break;
                }
                else {
                    let length = symbol - 254;
                    if (symbol > 264) {
                        const index = symbol - 257;
                        const lengthBits = fixedLengthExtraBits[index];
                        length = bits(input, position, (1 << lengthBits) - 1) + lengthBase[index];
                        position += lengthBits;
                    }
                    const code = distanceMap[bits16(input, position) & maxDistanceBitsMask];
                    if (!code) {
                        throw new zip.Error('Invalid distance.');
                    }
                    const distanceSymbol = code >>> 4;
                    position += code & 15;
                    let distance = distanceBase[distanceSymbol];
                    if (distanceSymbol > 3) {
                        const distanceBits = fixedDistanceExtraBits[distanceSymbol];
                        distance += bits16(input, position) & ((1 << distanceBits) - 1);
                        position += distanceBits;
                    }
                    if (position > inputBitLength) {
                        throw new zip.Error('Unexpected end of file.');
                    }
                    if (allocate) {
                        resize(offset + 131072);
                    }
                    const end = offset + length;
                    for (; offset < end; offset += 4) {
                        output[offset] = output[offset - distance];
                        output[offset + 1] = output[offset + 1 - distance];
                        output[offset + 2] = output[offset + 2 - distance];
                        output[offset + 3] = output[offset + 3 - distance];
                    }
                    offset = end;
                }
            }
            if (lengthMap) {
                final = 1;
            }
        } while (!final);
        if (offset !== output.length) {
            const buffer = new Uint8Array(offset);
            buffer.set(output.subarray(0, offset));
            return buffer;
        }
        return output;
    }

    static _initialize() {
        if (zip.Inflater._reverseMap) {
            return;
        }
        zip.Inflater._fixedLengthExtraBits = new Uint8Array([ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0, 0, 0 ]);
        zip.Inflater._fixedDistanceExtraBits = new Uint8Array([ 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 0, 0 ]);
        zip.Inflater._lengthBase = [ 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 260, 261 ];
        zip.Inflater._distanceBase = [ 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 32769 ];
        zip.Inflater._codeLengthIndexMap = new Uint8Array([ 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 ]);
        const reverseMap = new Uint16Array(32768);
        for (let i = 0; i < 32768; ++i) {
            let x = ((i & 0xaaaa) >>> 1) | ((i & 0x5555) << 1);
            x = ((x & 0xcccc) >>> 2) | ((x & 0x3333) << 2);
            x = ((x & 0xf0f0) >>> 4) | ((x & 0x0f0f) << 4);
            reverseMap[i] = (((x & 0xff00) >>> 8) | ((x & 0x00ff) << 8)) >>> 1;
        }
        zip.Inflater._reverseMap = reverseMap;
        const fixedLengthTree = new Uint8Array(288);
        for (let i = 0; i < 144; ++i) {
            fixedLengthTree[i] = 8;
        }
        for (let i = 144; i < 256; ++i) {
            fixedLengthTree[i] = 9;
        }
        for (let i = 256; i < 280; ++i) {
            fixedLengthTree[i] = 7;
        }
        for (let i = 280; i < 288; ++i) {
            fixedLengthTree[i] = 8;
        }
        const fixedDistanceTree = new Uint8Array(32);
        for (let i = 0; i < 32; ++i) {
            fixedDistanceTree[i] = 5;
        }
        zip.Inflater._fixedLengthMap = zip.Inflater._huffman(fixedLengthTree, 9);
        zip.Inflater._fixedDistanceMap = zip.Inflater._huffman(fixedDistanceTree, 5);
    }

    static _huffman(cd, maxBits) {
        const reverseMap = zip.Inflater._reverseMap;
        const s = cd.length;
        const l = new Uint16Array(maxBits);
        for (let i = 0; i < s; i++) {
            ++l[cd[i] - 1];
        }
        const le = new Uint16Array(maxBits);
        for (let i = 0; i < maxBits; i++) {
            le[i] = (le[i - 1] + l[i - 1]) << 1;
        }
        const co = new Uint16Array(1 << maxBits);
        const rvb = 15 - maxBits;
        for (let i = 0; i < s; i++) {
            if (cd[i]) {
                const sv = (i << 4) | cd[i];
                const r_1 = maxBits - cd[i];
                let v = le[cd[i] - 1]++ << r_1;
                for (let m = v | ((1 << r_1) - 1); v <= m; ++v) {
                    co[reverseMap[v] >>> rvb] = sv;
                }
            }
        }
        return co;
    }
};

zip.InflaterStream = class {

    constructor(stream, length) {
        this._stream = stream;
        this._position = 0;
        this._length = length;
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

    peek(length) {
        const position = this._position;
        length = length !== undefined ? length : this._length - position;
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
        length = length !== undefined ? length : this._length - position;
        this.skip(length);
        if (position === 0 && length === this._length) {
            return this._buffer;
        }
        return this._buffer.subarray(position, this._position);
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._buffer[position];
    }

    _inflate() {
        if (this._buffer === undefined) {
            const compressed = this._stream.peek();
            this._buffer = new Uint8Array(this._length);
            this._buffer = new zip.Inflater().inflateRaw(compressed, this._buffer);
            if (this._length != this._buffer.length) {
                throw new zip.Error('Invalid uncompressed size.');
            }
            delete this._stream;
        }
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
};

zip.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Zip Error';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Archive = zip.Archive;
    module.exports.Inflater = zip.Inflater;
}