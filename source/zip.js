/* jshint esversion: 6 */
/* global pako */

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
            let position = stream.length;
            do {
                position = Math.max(0, position - 64000);
                stream.seek(position);
                const buffer = stream.read(Math.min(stream.length - position, 65536));
                for (let i = buffer.length - 4; i >= 0; i--) {
                    if (signature[0] === buffer[i] && signature[1] === buffer[i + 1] && signature[2] === buffer[i + 2] && signature[3] === buffer[i + 3]) {
                        stream.seek(position + i + 4);
                        return true;
                    }
                }
            }
            while (position > 0);
            return false;
        };
        if (!lookup(stream, [ 0x50, 0x4B, 0x05, 0x06 ])) {
            throw new zip.Error('End of central directory not found.');
        }
        let reader = new zip.BinaryReader(stream.read(16));
        reader.skip(12);
        let offset = reader.uint32(); // central directory offset
        if (offset > stream.length) {
            console.log("*");
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

    inflateRaw(data, length) {
        let buffer = null;
        if (typeof process === 'object' && typeof process.versions == 'object' && typeof process.versions.node !== 'undefined') {
            buffer = require('zlib').inflateRawSync(data);
        }
        else if (typeof pako !== 'undefined') {
            buffer = pako.inflateRaw(data);
        }
        else {
            const reader = new zip.BitReader(data);
            const writer = length === undefined ? new zip.BlockWriter() : new zip.BufferWriter(length);

            if (!zip.Inflater._staticLengthTree) {

                zip.Inflater._codeOrder = [ 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 ];
                zip.Inflater._codeTree = new zip.HuffmanTree();
                zip.Inflater._lengths = new Uint8Array(288 + 32);
                zip.Inflater._lengthBits = [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 6 ];
                zip.Inflater._lengthBase = [ 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 323 ];
                zip.Inflater._distanceBits = [ 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 ];
                zip.Inflater._distanceBase = [ 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 ];

                zip.Inflater._staticLengthTree = new zip.HuffmanTree();
                zip.Inflater._staticLengthTree.table = new Uint8Array([ 0, 0, 0, 0, 0,  0, 0, 24, 152, 112, 0, 0, 0, 0, 0, 0 ]);
                for (let i = 0; i < 24; ++i) {
                    zip.Inflater._staticLengthTree.symbol[i] = 256 + i;
                }
                for (let i = 0; i < 144; ++i) {
                    zip.Inflater._staticLengthTree.symbol[24 + i] = i;
                }
                for (let i = 0; i < 8; ++i) {
                    zip.Inflater._staticLengthTree.symbol[24 + 144 + i] = 280 + i;
                }
                for (let i = 0; i < 112; ++i) {
                    zip.Inflater._staticLengthTree.symbol[24 + 144 + 8 + i] = 144 + i;
                }
                zip.Inflater._staticDistanceTree = new zip.HuffmanTree();
                zip.Inflater._staticDistanceTree.table = new Uint8Array([ 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
                zip.Inflater._staticDistanceTree.symbol = new Uint8Array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ]);
            }

            let type;
            do {
                type = reader.bits(3);
                switch (type >>> 1) {
                    case 0: { // uncompressed block
                        this._copyUncompressedBlock(reader, writer);
                        break;
                    }
                    case 1: { // block with fixed huffman trees
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
                        throw new zip.Error('Unknown block type.');
                    }
                }
            } while ((type & 1) == 0);
            if (length !== undefined && length !== writer.length) {
                throw new zip.Error('Invalid uncompressed size.');
            }
            buffer = writer.toBuffer();
        }
        if (length !== undefined && length !== buffer.length) {
            throw new zip.Error('Invalid uncompressed size.');
        }
        return buffer;
    }

    _copyUncompressedBlock(reader, writer) {
        reader.position = (reader.position + 7) & ~7; // align
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
        const lengthCount = reader.bits(4) + 4;

        const lengths = zip.Inflater._lengths;
        for (let i = 0; i < 19; i++) {
            lengths[i] = 0;
        }
        for (let j = 0; j < lengthCount; j++) {
            lengths[zip.Inflater._codeOrder[j]] = reader.bits(3);
        }
        zip.Inflater._codeTree.build(lengths, 0, 19);
        for (let i = 0; i < hlit + hdist;) {
            const symbol = reader.symbol(zip.Inflater._codeTree);
            switch (symbol) {
                case 16: {
                    const prev = lengths[i - 1];
                    for (let j = reader.bits(2) + 3; j; j--) {
                        lengths[i++] = prev;
                    }
                    break;
                }
                case 17: {
                    for (let j = reader.bits(3) + 3; j > 0; j--) {
                        lengths[i++] = 0;
                    }
                    break;
                }
                case 18: {
                    for (let j = reader.bits(7) + 11; j > 0; j--) {
                        lengths[i++] = 0;
                    }
                    break;
                }
                default: {
                    lengths[i++] = symbol;
                    break;
                }
            }
        }

        this._lengthTree = new zip.HuffmanTree();
        this._lengthTree.build(zip.Inflater._lengths, 0, hlit);
        this._distanceTree = new zip.HuffmanTree();
        this._distanceTree.build(zip.Inflater._lengths, hlit, hdist);
    }

    _inflateBlock(reader, writer) {
        const lengthTree = this._lengthTree;
        const lengthBits = zip.Inflater._lengthBits;
        const lengthBase = zip.Inflater._lengthBase;
        const distanceTree = this._distanceTree;
        const distanceBits = zip.Inflater._distanceBits;
        const distanceBase = zip.Inflater._distanceBase;
        const buffer = writer.buffer;
        const threshold = writer.threshold !== undefined ? writer.threshold : writer.length;
        let position = writer.position;
        for (;;) {
            if (position > threshold) {
                position = writer.push(position);
            }
            let symbol = reader.symbol(lengthTree);
            if (symbol < 256) {
                buffer[position++] = symbol;
            }
            else if (symbol === 256) {
                writer.push(position);
                return;
            }
            else {
                symbol -= 257;
                const length = reader.bits16(lengthBits[symbol]) + lengthBase[symbol];
                const distance = reader.symbol(distanceTree);
                let offset = position - reader.bits16(distanceBits[distance]) - distanceBase[distance];
                for (let i = 0; i < length; i++) {
                    buffer[position++] = buffer[offset++];
                }
            }
        }
    }
};

zip.HuffmanTree = class {

    constructor() {
        this.table = new Uint16Array(16);
        this.symbol = new Uint16Array(288);
        zip.HuffmanTree._offsets = zip.HuffmanTree._offsets || new Uint16Array(16);
    }

    build(lengths, offset, count) {
        for (let i = 0; i < 16; ++i) {
            this.table[i] = 0;
        }
        for (let i = 0; i < count; ++i) {
            this.table[lengths[offset + i]]++;
        }
        this.table[0] = 0;
        let sum = 0;
        for (let i = 0; i < 16; i++) {
            zip.HuffmanTree._offsets[i] = sum;
            sum += this.table[i];
        }
        for (let i = 0; i < count; i++) {
            if (lengths[offset + i]) {
                this.symbol[zip.HuffmanTree._offsets[lengths[offset + i]]++] = i;
            }
        }
    }
};

zip.BitReader = class {

    constructor(buffer) {
        this.buffer = buffer;
        this.position = 0;
    }

    bits(count) {
        const mask = (1 << count) - 1;
        const offset = (this.position / 8) >> 0;
        const value = ((this.buffer[offset] | (this.buffer[offset + 1] << 8)) >>> (this.position & 7)) & mask;
        this.position += count;
        return value;
    }

    bits16(count) {
        const mask = (1 << count) - 1;
        const offset = (this.position / 8) >> 0;
        const value = ((this.buffer[offset] | (this.buffer[offset + 1] << 8) | (this.buffer[offset + 2] << 16)) >>> (this.position & 7)) & mask;
        this.position += count;
        return value;
    }

    read(size) {
        const offset = (this.position / 8) >> 0;
        const value = this.buffer.subarray(offset, offset + size);
        this.position += size * 8;
        return value;
    }

    uint16() {
        const offset = (this.position / 8) >> 0;
        const value = this.buffer[offset] | (this.buffer[offset + 1] << 8);
        this.position += 16;
        return value;
    }

    symbol(tree) {
        let sum = 0;
        let current = 0;
        let length = 0;
        const table = tree.table;
        do {
            current = (current << 1) + this.bits(1);
            length++;
            sum += table[length];
            current -= table[length];
        } while (current >= 0);
        return tree.symbol[sum + current];
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
        }
        else {
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
            const buffer = this._stream.peek();
            this._buffer = new zip.Inflater().inflateRaw(buffer, this._length);
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