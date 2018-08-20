/*jshint esversion: 6 */

var zip = zip || {};

zip.Archive = class {

    // inflateRaw (optional): optimized inflate callback. For example, require('zlib').inflateRawSync or pako.inflateRaw.
    constructor(buffer, inflateRaw) {
        this._entries = [];
        if (buffer.length < 4 || buffer[0] != 0x50 || buffer[1] != 0x4B) {
            throw new zip.Error('Invalid ZIP archive.');
        }
        var reader = null;
        for (var i = buffer.length - 4; i >= 0; i--) {
            if (buffer[i] === 0x50 && buffer[i + 1] === 0x4B && buffer[i + 2] === 0x05 && buffer[i + 3] === 0x06) {
                reader = new zip.Reader(buffer, i + 4, buffer.length);
                break;
            }
        }
        if (!reader) {
            throw new zip.Error('End of central directory not found.');
        }
        reader.skip(12);
        reader.position = reader.readUint32(); // central directory offset
        while (reader.match([ 0x50, 0x4B, 0x01, 0x02 ])) {
            this._entries.push(new zip.Entry(reader, inflateRaw));
        }
    }

    get entries() {
        return this._entries;
    }
};

zip.Entry = class {

    constructor(reader, inflateRaw) {
        this._inflateRaw = inflateRaw;
        reader.readUint16(); // version
        reader.skip(2);
        this._flags = reader.readUint16();
        if ((this._flags & 1) == 1) {
            throw new zip.Error('Encrypted entries not supported.');
        }
        this._compressionMethod = reader.readUint16();
        reader.readUint32(); // date
        reader.readUint32(); // crc32
        this._compressedSize = reader.readUint32();
        this._size = reader.readUint32();
        var nameLength = reader.readUint16(); // file name length
        var extraDataLength = reader.readUint16();
        var commentLength = reader.readUint16();
        reader.readUint16(); // disk number start
        reader.readUint16(); // internal file attributes
        reader.readUint32(); // external file attributes
        var localHeaderOffset = reader.readUint32();
        reader.skip(nameLength);
        reader.skip(extraDataLength);
        reader.read(commentLength); // comment
        var position = reader.position;
        reader.position = localHeaderOffset;
        if (!reader.match([ 0x50, 0x4B, 0x03, 0x04 ])) {
            throw new zip.Error('Invalid local file header signature.');
        }
        reader.skip(22);
        nameLength = reader.readUint16();
        extraDataLength = reader.readUint16();
        var nameBuffer = reader.read(nameLength);
        this._name = '';
        nameBuffer.forEach((c) => {
            this._name += String.fromCharCode(c);
        });
        reader.skip(extraDataLength);
        this._compressedData = reader.read(this._compressedSize);
        reader.position = position;
    }

    get name() {
        return this._name;
    }

    get data() {
        if (!this._data) {

            switch (this._compressionMethod) {
                case 0: // Stored
                    if (this._size != this._compressedSize) {
                        throw new zip.Error('Invalid compression size.');
                    }
                    this._data = new Uint8Array(this._compressedData.length);
                    this._data.set(this._compressedData);
                    break;
                case 8: // Deflate
                    if (this._inflateRaw) {
                        this._data = this._inflateRaw(this._compressedData);
                    }
                    else {
                        this._data = new zip.Inflater().inflateRaw(this._compressedData);
                    }
                    if (this._size != this._data.length) {
                        throw new zip.Error('Invalid uncompressed size.');
                    }
                    break;
                default:
                    throw new zip.Error('Invalid compression method.');
            }

            delete this._size;
            delete this._inflateRaw;
            delete this._compressedData;
        }
        return this._data;
    }

};

zip.HuffmanTree = class {

    constructor() {
        this.table = new Uint16Array(16);
        this.symbol = new Uint16Array(288);
        zip.HuffmanTree._offsets = zip.HuffmanTree._offsets || new Uint16Array(16);
    }

    build(lengths, offset, count) {
        var i;
        for (i = 0; i < 16; ++i) {
            this.table[i] = 0;
        }
        for (i = 0; i < count; ++i) {
            this.table[lengths[offset + i]]++;
        }
        this.table[0] = 0;
        var sum = 0;
        for (i = 0; i < 16; i++) {
            zip.HuffmanTree._offsets[i] = sum;
            sum += this.table[i];
        }
        for (i = 0; i < count; i++) {
            if (lengths[offset + i]) {
                this.symbol[zip.HuffmanTree._offsets[lengths[offset + i]]++] = i;
            }
        }
    }

    static initialize() {
        if (!zip.HuffmanTree.staticLiteralLengthTree) {
            var i;
            zip.HuffmanTree.staticLiteralLengthTree = new zip.HuffmanTree();
            zip.HuffmanTree.staticLiteralLengthTree.table = new Uint8Array([ 0, 0, 0, 0, 0,  0, 0, 24, 152, 112, 0, 0, 0, 0, 0, 0 ]);
            for (i = 0; i < 24; ++i) { 
                zip.HuffmanTree.staticLiteralLengthTree.symbol[i] = 256 + i;
            }
            for (i = 0; i < 144; ++i) {
                zip.HuffmanTree.staticLiteralLengthTree.symbol[24 + i] = i;
            }
            for (i = 0; i < 8; ++i) {
                zip.HuffmanTree.staticLiteralLengthTree.symbol[24 + 144 + i] = 280 + i;
            }
            for (i = 0; i < 112; ++i) {
                zip.HuffmanTree.staticLiteralLengthTree.symbol[24 + 144 + 8 + i] = 144 + i;
            }
            zip.HuffmanTree.staticDistanceTree = new zip.HuffmanTree();
            zip.HuffmanTree.staticDistanceTree.table = new Uint8Array([ 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
            zip.HuffmanTree.staticDistanceTree.symbol = new Uint8Array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ]);
        }
    }
};

zip.Inflater = class {

    inflateRaw(data) {

        zip.Inflater.initilize();
        zip.HuffmanTree.initialize();

        var reader = new zip.BitReader(data);
        var output = new zip.Ouptut();

        var literalLengthTree = new zip.HuffmanTree();
        var distanceTree = new zip.HuffmanTree();

        var type;
        do {
            type = reader.readBits(3);
            switch (type >>> 1) {
                case 0: // uncompressed block
                    this._inflateUncompressedBlock(reader, output);
                    break;
                case 1: // block with fixed huffman trees
                    this._inflateBlockData(reader, output, zip.HuffmanTree.staticLiteralLengthTree, zip.HuffmanTree.staticDistanceTree);
                    break;
                case 2: // block with dynamic huffman trees
                    this._decodeTrees(reader, literalLengthTree, distanceTree);
                    this._inflateBlockData(reader, output, literalLengthTree, distanceTree);
                    break;
                default:
                    throw new Error('Unknown block type.');
            }
        } while ((type & 1) == 0);

        return output.merge();
    }

    _inflateUncompressedBlock(reader, output) {
        while (reader.bits > 8) {
            reader.position--;
            reader.bits -= 8;
        }
        reader.bits = 0;
        var length = reader.readUint16(); 
        var inverseLength = reader.readUint16();
        if (length !== (~inverseLength & 0x0000ffff)) {
            throw new Error('Invalid uncompressed block length.');
        }

        var block = reader.read(length);
        output.push(block);
 
        if (length > 32768) {
            output.buffer.set(block.subarray(block.length - 32768, block.length), 0);
            output.position = 32768;
        }
        else {
            output.reset();
            output.buffer.set(block, output.position);
            output.position += block.length;
        }
    }

    _decodeTrees(reader, lengthTree, distanceTree) {

        var hlit = reader.readBits(5) + 257;
        var hdist = reader.readBits(5) + 1;
        var lengthCount = reader.readBits(4) + 4;
        for (var i = 0; i < 19; i++) {
            zip.Inflater._lengths[i] = 0;
        }
        for (var j = 0; j < lengthCount; j++) {
            zip.Inflater._lengths[zip.Inflater._codeOrder[j]] = reader.readBits(3);
        }
        zip.Inflater._codeTree.build(zip.Inflater._lengths, 0, 19);
        var length;  
        for (var position = 0; position < hlit + hdist;) {
            var symbol = reader.readSymbol(zip.Inflater._codeTree);
            switch (symbol) {
                case 16:
                    var prev = zip.Inflater._lengths[position - 1];
                    for (length = reader.readBits(2) + 3; length; length--) {
                        zip.Inflater._lengths[position++] = prev;
                    }
                    break;
                case 17:
                    for (length = reader.readBits(3) + 3; length; length--) {
                        zip.Inflater._lengths[position++] = 0;
                    }
                    break;
                case 18:
                    for (length = reader.readBits(7) + 11; length; length--) {
                        zip.Inflater._lengths[position++] = 0;
                    }
                    break;
                default:
                    zip.Inflater._lengths[position++] = symbol;
                    break;
            }
        }
        lengthTree.build(zip.Inflater._lengths, 0, hlit);
        distanceTree.build(zip.Inflater._lengths, hlit, hdist);
    }

    _inflateBlockData(reader, output, lengthTree, distanceTree) {
        var buffer = output.buffer;
        var position = output.position;
        var start = position;
        while (true) {
            if (position > 62464) {
                output.position = position;
                output.push(new Uint8Array(buffer.subarray(start, position)));
                position = output.reset();
                start = position;
            }
            var symbol = reader.readSymbol(lengthTree);
            if (symbol === 256) {
                output.position = position;
                output.push(new Uint8Array(buffer.subarray(start, output.position)));
                output.reset();
                return;
            }
            if (symbol < 256) {
                buffer[position++] = symbol;
            }
            else {
                symbol -= 257;
                var length = reader.readBitsBase(zip.Inflater._lengthBits[symbol], zip.Inflater._lengthBase[symbol]);
                var distance = reader.readSymbol(distanceTree);
                var offset = position - reader.readBitsBase(zip.Inflater._distanceBits[distance], zip.Inflater._distanceBase[distance]);
                for (var i = 0; i < length; i++) {
                    buffer[position++] = buffer[offset++];
                }
            }
        }
    }

    static initilize() {
        if (zip.HuffmanTree.staticLiteralLengthTree) {
            return;
        }
        zip.Inflater._codeOrder = [ 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 ];
        zip.Inflater._codeTree = new zip.HuffmanTree();
        zip.Inflater._lengths = new Uint8Array(288 + 32);
        zip.Inflater._lengthBits = [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 6 ];
        zip.Inflater._lengthBase = [ 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 323 ];
        zip.Inflater._distanceBits = [ 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 ];
        zip.Inflater._distanceBase = [ 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 ];
    }

};

zip.Ouptut = class {

    constructor() {
        this._blocks = [];
        this.buffer = new Uint8Array(65536);
        this.position = 0;
    }

    reset() {
        if (this.position > 32768) {
            this.buffer.set(this.buffer.subarray(this.position - 32768, this.position), 0);
            this.position = 32768;
        }
        return this.position;
    }

    push(block) {
        this._blocks.push(block);
    }

    merge() {
        var size = 0;
        this._blocks.forEach((block) => {
            size += block.length;
        });
        var output = new Uint8Array(size);
        var offset = 0;
        this._blocks.forEach((block) => {
            output.set(block, offset);
            offset += block.length;
        });
        return output;
    }

};

zip.BitReader = class {

    constructor(buffer) {
        this.buffer = buffer;
        this.position = 0;
        this.bits = 0;
        this.value = 0;
    }

    readBits(count) {
        while (this.bits < 24) {
            this.value |= this.buffer[this.position++] << this.bits;
            this.bits += 8;
        }
        var value = this.value & (0xffff >>> (16 - count));
        this.value >>>= count;
        this.bits -= count;
        return value;
    }

    readBitsBase(count, base) {
        if (count == 0) {
            return base;
        }
        while (this.bits < 24) {
            this.value |= this.buffer[this.position++] << this.bits;
            this.bits += 8;
        }
        var value = this.value & (0xffff >>> (16 - count));
        this.value >>>= count;
        this.bits -= count;
        return value + base;
    }

    read(size) {
        var value = this.buffer.subarray(this.position, this.position + size);
        reader.position += size;
        return value;
    }

    readUint16() {
        var value = this.buffer[this.position] | (this.buffer[this.position + 1] << 8);
        this.position += 2;
        return value;
    }

    readSymbol(tree) {
        while (this.bits < 24) {
            this.value |= this.buffer[this.position++] << this.bits;
            this.bits += 8;
        }
        var sum = 0;
        var current = 0;
        var length = 0;
        var value = this.value;
        var table = tree.table;
        do {
            current = (current << 1) + (value & 1);
            value >>>= 1;
            length++;
            sum += table[length];
            current -= table[length];
        } while (current >= 0);
        this.value = value;
        this.bits -= length;
        return tree.symbol[sum + current];
    }

};

zip.Reader = class {

    constructor(buffer, start, end) {
        this._buffer = buffer;
        this._position = start;
        this._end = end;
    }

    match(signature) {
        if (this._position + signature.length <= this._end) {
            for (var i = 0; i < signature.length; i++) {
                if (this._buffer[this._position + i] != signature[i]) {
                    return false;
                }
            }
        }
        this._position += signature.length;
        return true;
    }

    get position() {
        return this._position;
    }

    set position(value) {
        this._position = value >= 0 ? value : this._end + value;
    }

    peek() {
        return this._position < this._end;
    }

    skip(size) {
        if (this._position + size > this._end) {
            throw new zip.Error('Data not available.');
        }
        this._position += size;
    }

    read(size) {
        if (this._position + size > this._end) {
            throw new zip.Error('Data not available.');
        }
        size = size === undefined ? this._end : size;
        var data = this._buffer.subarray(this._position, this._position + size);
        this._position += size;
        return data;
    }

    readByte() {
        if (this._position + 1 > this._end) {
            throw new zip.Error('Data not available.');
        }
        var value = this._buffer[this._position];
        this._position++;
        return value;
    }

    readUint16() {
        if (this._position + 2 > this._end) {
            throw new zip.Error('Data not available.');
        }
        var value = this._buffer[this._position] | (this._buffer[this._position + 1] << 8);
        this._position += 2;
        return value;
    }

    readUint32() {
        return this.readUint16() | (this.readUint16() << 16);
    }

    readString() {
        var result = '';
        var end = this._buffer.indexOf(0x00, this._position);
        if (end < 0) {
            throw new zip.Error('End of string not found.');
        }
        while (this._position < end) {
            result += String.fromCharCode(this._buffer[this._position++]);
        }
        this._position++;
        return result;
    }

};

zip.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'ZIP Error';
    }
};

var gzip = gzip || {};

gzip.Archive = class {

    // inflate (optional): optimized inflater callback like require('zlib').inflateRawSync or pako.inflateRa
    constructor(buffer, inflateRaw) {
        this._entries = [];
        if (buffer.length < 18 || buffer[0] != 0x1f || buffer[1] != 0x8b) {
            throw new zip.Error('Invalid GZIP archive.');
        }
        var reader = new zip.Reader(buffer, 0, buffer.length);
        this._entries.push(new gzip.Entry(reader, inflateRaw));
    }

    get entries() {
        return this._entries;
    }
};

gzip.Entry = class {

    constructor(reader, inflateRaw) {
        if (!reader.match([ 0x1f, 0x8b ])) {
            throw new zip.Error('Invalid GZIP signature.');
        }
        var compressionMethod = reader.readByte();
        if (compressionMethod != 8) {
            throw new zip.Error('Invalid compression method ' + compressionMethod.toString() + "'.");
        }
        var flags = reader.readByte();
        reader.readUint32(); // MTIME
        var xflags = reader.readByte();
        reader.readByte(); // OS
        if ((flags & 4) != 0) {
            var xlen = reader.readUint16();
            reader.skip(xlen);
        }
        if ((flags & 8) != 0) {
            this._name = reader.readString();
        }
        if ((flags & 16) != 0) { // FLG.FCOMMENT
            reader.readString();
        }
        if ((flags & 1) != 0) {
            reader.readUint16(); // CRC16
        }
        var compressedData = reader.read();
        if (inflateRaw) {
            this._data = inflateRaw(compressedData);
        }
        else {
            this._data = new zip.Inflater().inflateRaw(compressedData);
        }
        reader.position = -8;
        reader.readUint32(); // CRC32
        var size = reader.readUint32();
        if (size != this._data.length) {
            throw new zip.Error('Invalid size.');
        }
    }

    get name() {
        return this._name;
    }

    get data() {
        return this._data;
    }

};
