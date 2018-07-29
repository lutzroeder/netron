/*jshint esversion: 6 */

var zip = zip || {};

zip.Archive = class {

    constructor(buffer, inflate) {
        this._inflate = inflate; // (optional) optimized inflater like require('zlib').inflateRawSync or pako.inflateRaw
        this._entries = [];
        for (var i = buffer.length - 4; i >= 0; i--) {
            if (buffer[i] === 0x50 && buffer[i + 1] === 0x4B && buffer[i + 2] === 0x05 && buffer[i + 3] === 0x06) {
                this._reader = new zip.Reader(buffer, i + 4, buffer.length);
                break;
            }
        }
        if (!this._reader) {
            throw new zip.Error('End of central directory not found.');
        }
        this._reader.skip(12);
        this._reader.position = this._reader.readUint32(); // central directory offset
        while (this._reader.checkSignature([ 0x50, 0x4B, 0x01, 0x02 ])) {
            this._entries.push(new zip.Entry(this._reader, inflate));
        }
    }

    get entries() {
        return this._entries;
    }
};

zip.Entry = class {

    constructor(reader, inflate) {
        this._inflate = inflate;
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
        if (!reader.checkSignature([ 0x50, 0x4B, 0x03, 0x04 ])) {
            throw new zip.Error('Invalid local file header signature.');
        }
        reader.skip(22);
        nameLength = reader.readUint16();
        extraDataLength = reader.readUint16();
        var nameBuffer = reader.read(nameLength);
        this._name = new TextDecoder('ascii').decode(nameBuffer);
        reader.skip(extraDataLength);
        this._reader = reader.readReader(this._compressedSize);
        reader.position = position;
    }

    get name() {
        return this._name;
    }

    get data() {
        if (!this._data) {

            var compressedData = this._reader.read(this._compressedSize);
            this._reader = null;

            switch (this._compressionMethod) {
                case 0: // Stored
                    if (this._size != this._compressedSize) {
                        throw new zip.Error('Invalid compression size.');
                    }
                    this._data = compressedData;
                    break;
                case 8: // Deflate
                    if (this._inflate) {
                        this._data = this._inflate(compressedData);
                    }
                    else {
                        var data = new Uint8Array(this._size);
                        var inflater = new zip.Inflater(compressedData, data);
                        inflater.inflate();
                        this._data = data;
                    }
                    break;
                default:
                    throw new zip.Error('Invalid compression method.');
            }
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
};

zip.Inflater = class {

    constructor(input, output) {
        this._input = input;
        this._inputPosition = 0;
        this._output = output;
        this._outputPosition = 0;

        this._bits = 0;
        this._value = 0;

        this._literalLengthTree = new zip.HuffmanTree();
        this._distanceTree = new zip.HuffmanTree();

        if (zip.HuffmanTree.staticLiteralLengthTree) {
            return;
        }
        var i;
        zip.HuffmanTree.staticLiteralLengthTree = new zip.HuffmanTree();
        zip.HuffmanTree.staticLiteralLengthTree.table = new Uint8Array([ 0, 0, 0, 0, 0,  0, 0, 24, 152, 112, 0, 0, 0, 0, 0, 0 ]);
        for (i = 0; i < 24; ++i) { zip.HuffmanTree.staticLiteralLengthTree.symbol[i] = 256 + i; }
        for (i = 0; i < 144; ++i) { zip.HuffmanTree.staticLiteralLengthTree.symbol[24 + i] = i; }
        for (i = 0; i < 8; ++i) { zip.HuffmanTree.staticLiteralLengthTree.symbol[24 + 144 + i] = 280 + i; }
        for (i = 0; i < 112; ++i) { zip.HuffmanTree.staticLiteralLengthTree.symbol[24 + 144 + 8 + i] = 144 + i; }
        zip.HuffmanTree.staticDistanceTree = new zip.HuffmanTree();
        zip.HuffmanTree.staticDistanceTree.table = new Uint8Array([ 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
        for (i = 0; i < 32; ++i) { zip.HuffmanTree.staticDistanceTree.symbol[i] = i; }
        zip.Inflater._codeOrder = [ 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 ];
        zip.Inflater._codeTree = new zip.HuffmanTree();
        zip.Inflater._lengths = new Uint8Array(288 + 32);
        zip.Inflater._lengthBits = [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 6 ];
        zip.Inflater._lengthBase = [ 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 323 ];
        zip.Inflater._distanceBits = [ 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 ];
        zip.Inflater._distanceBase = [ 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 ];
    }

    inflate() {
        var type;
        do {
            type = this._readBits(3);
            switch (type >>> 1) {
                case 0: // uncompressed block
                    this._inflateUncompressedBlock();
                    break;
                case 1: // block with fixed huffman trees
                    this._inflateBlockData(zip.HuffmanTree.staticLiteralLengthTree, zip.HuffmanTree.staticDistanceTree);
                    break;
                case 2: // block with dynamic huffman trees
                    this._decodeTrees(this._literalLengthTree, this._distanceTree);
                    this._inflateBlockData(this._literalLengthTree, this._distanceTree);
                    break;
                default:
                    throw new Error('Invalid block.');
            }
        } while ((type & 1) == 0);
        if (this._outputPosition != this._output.length) {
            throw new zip.Error('Invalid uncompressed size.');
        }
        return this._output;
    }

    _readBits(count) {
        while (this._bits < 24) {
            this._value |= this._input[this._inputPosition++] << this._bits;
            this._bits += 8;
        }
        var value = this._value & (0xffff >>> (16 - count));
        this._value >>>= count;
        this._bits -= count;
        return value;
    }

    _readBitsBase(count, base) {
        if (count == 0) {
            return base;
        }
        while (this._bits < 24) {
            this._value |= this._input[this._inputPosition++] << this._bits;
            this._bits += 8;
        }
        var value = this._value & (0xffff >>> (16 - count));
        this._value >>>= count;
        this._bits -= count;
        return value + base;
    }

    _readSymbol(tree) {
        while (this._bits < 24) {
            this._value |= this._input[this._inputPosition++] << this._bits;
            this._bits += 8;
        }
        var sum = 0;
        var current = 0;
        var length = 0;
        var value = this._value;
        do {
            current = (current << 1) + (value & 1);
            value >>>= 1;
            length++;
            sum += tree.table[length];
            current -= tree.table[length];
        } while (current >= 0);
        this._value = value;
        this._bits -= length;
        return tree.symbol[sum + current];
    }

    _inflateUncompressedBlock() {
        while (this._bits > 8) {
            this._inputPosition--;
            this._bits -= 8;
        }
        var length = (this._input[this._inputPosition + 1] << 8) | this._input[this._inputPosition]; 
        var invlength = (this._input[this._inputPosition + 3] << 8) | this._input[this._inputPosition + 2];
        if (length !== (~invlength & 0x0000ffff)) {
            throw new Error('Invalid uncompressed block length.');
        }
        this._inputPosition += 4;
        for (var i = length; i; --i) {
            this._output[d.destLen++] = this._input[d.sourceIndex++];
        }
        this._bits = 0;
    }

    _decodeTrees(lengthTree, distanceTree) {
        var hlit = this._readBits(5) + 257;
        var hdist = this._readBits(5) + 1;
        var lengthCount = this._readBits(4) + 4;
        for (var i = 0; i < 19; i++) {
            zip.Inflater._lengths[i] = 0;
        }
        for (var j = 0; j < lengthCount; j++) {
            zip.Inflater._lengths[zip.Inflater._codeOrder[j]] = this._readBits(3);
        }
        zip.Inflater._codeTree.build(zip.Inflater._lengths, 0, 19);
        var length;  
        for (var position = 0; position < hlit + hdist;) {
            var symbol = this._readSymbol(zip.Inflater._codeTree);
            switch (symbol) {
                case 16:
                    var prev = zip.Inflater._lengths[position - 1];
                    for (length = this._readBits(2) + 3; length; length--) {
                        zip.Inflater._lengths[position++] = prev;
                    }
                    break;
                case 17:
                    for (length = this._readBits(3) + 3; length; length--) {
                        zip.Inflater._lengths[position++] = 0;
                    }
                    break;
                case 18:
                    for (length = this._readBits(7) + 11; length; length--) {
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

    _inflateBlockData(lengthTree, distanceTree) {
        while (true) {
            var symbol = this._readSymbol(lengthTree);
            if (symbol === 256) {
                return;
            }
            if (symbol < 256) {
                this._output[this._outputPosition++] = symbol;
            }
            else {
                symbol -= 257;
                var length = this._readBitsBase(zip.Inflater._lengthBits[symbol], zip.Inflater._lengthBase[symbol]);
                var distance = this._readSymbol(distanceTree);
                var offset = this._outputPosition - this._readBitsBase(zip.Inflater._distanceBits[distance], zip.Inflater._distanceBase[distance]);
                for (var i = offset; i < offset + length; ++i) {
                    this._output[this._outputPosition++] = this._output[i];
                }
            }
        }
    }

};

zip.Reader = class {

    constructor(buffer, start, end) {
        this._buffer = buffer;
        this._position = start;
        this._end = end;
    }

    checkSignature(signature) {
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
        this._position = value;
    }

    readReader(size) {
        if (this._position + size > this._end) {
            throw new zip.Error('Data not available.');
        }
        var reader = new zip.Reader(this._buffer, this._position, this._position + size);
        this._position += size;
        return reader;
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
        var data = this._buffer.subarray(this._position, this._position + size);
        this._position += size;
        return data;
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
};

zip.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'ZIP Error';
    }
};
