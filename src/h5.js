/*jshint esversion: 6 */

// Experimental H5/HDF5 reader. Currently only supports reading <string,string> attributes. 

class H5 {

    constructor(buffer) {
        H5.utf8Decoder = new TextDecoder('utf-8');
        H5.asciiDecoder = new TextDecoder('ascii');

        var reader = new H5.Reader(buffer, 0);

        this._globalHeap = new H5.GlobalHeap(reader);

        var signature = [137, 72, 68, 70, 13, 10, 26, 10];
        while (signature.length > 0) {
            if (reader.readByte() != signature.shift()) { 
                throw new Error("Not a valid H5 file.");
            }
        }

        var version = reader.readByte();
        if (version == 0 || version == 1) {
            this._freeSpaceStorageVersion = reader.readByte();
            this._rootGroupSymbolTableEntryVersion = reader.readByte();
            reader.skipBytes(1);
            this._sharedHeaderMessageVersionFormat = reader.readByte();
            reader.setOffsetSize(reader.readByte());
            reader.setLengthSize(reader.readByte());
            reader.skipBytes(1);

            this._groupLeafNodeK = reader.readUint16(); // 0x04?
            this._groupInternalNodeK = reader.readUint16(); // 0x10?
    
            reader.skipBytes(4);
    
            if (version > 0) {
                this._indexedStorageInternalNodeK = reader.readUint16();
                this.skipBytes(2); // Reserved
            }
    
            this._baseAddress = reader.readOffset();
            reader.skipOffset(); // Address of File Free space Info
            this._endOfFileAddress = reader.readOffset();
            reader.skipOffset(); // Driver Information Block Address
            if (this._baseAddress != 0) {
                throw new Error('Base address is not zero.');
            }

            this._rootGroupSymbleTableEntry = new H5.SymbolTableEntry(reader, this._globalHeap);
        }
        else {
            throw new Error('Unknown Superblock version ' + version + '.');
        }
    }

    get rootGroup() {
        return this._rootGroupSymbleTableEntry;
    }
}

H5.Reader = class {
    constructor(buffer, position) {
        this._buffer = buffer;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._position = position;
    }

    readByte() {
        return this._dataView.getUint8(this._position++);
    }

    skipBytes(length) {
        this._position += length;
    }

    readBytes(length) {
        var data = this._buffer.subarray(this._position, this._position + length);
        this._position += length;
        return data;
    }

    readUint16() {
        var value = this._dataView.getUint16(this._position, true);
        this._position += 2;
        return value;
    }

    readUint32() {
        var value = this._dataView.getUint32(this._position, true);
        this._position += 4;
        return value;
    }

    readUint64() {
        var lo = this._dataView.getUint32(this._position, true);
        var hi = this._dataView.getUint32(this._position + 4, true);
        this._position += 8;
        if (lo == 4294967295 && hi == lo) {
            return -1;
        } 
        if (hi != 0) {
            throw new Error('File address outside 32-bit range.');
        }
        return lo;
    }

    setOffsetSize(size) {
        this._offsetSize = size;
    }

    readOffset() { 
        switch (this._offsetSize) {
            case 8:
               return this.readUint64();
            case 4:
                return this.readUint32(); 
        }
        throw new Error('Unknown offset size \'' + this._offsetSize + '\'.');
    }

    skipOffset() {
        this.skipBytes(this._offsetSize);
    }

    setLengthSize(size) {
        this._lengthSize = size;
    }

    readLength() {
        switch (this._lengthSize) {
            case 8:
               return this.readUint64();
            case 4:
                return this.readUint32(); 
        }
        throw new Error('Unknown length size \'' + this._lengthSize + '\'.');
    }

    skipLength() {
        this.skipBytes(this._lengthSize);
    }

    move(position) {
        var reader = new H5.Reader(this._buffer, position);
        reader.setOffsetSize(this._offsetSize);
        reader.setLengthSize(this._lengthSize);
        return reader;
    }

    clone() {
        var reader =  new H5.Reader(this._buffer, this._position);
        reader.setOffsetSize(this._offsetSize);
        reader.setLengthSize(this._lengthSize);
        return reader;
    }

    align(mod) {
        if (this._position % mod != 0) {
            this._position = (Math.floor(this._position / mod) + 1) * mod;
        }
    }

    get position() {
        return this._position;
    }
};

H5.SymbolTableEntry = class {
    constructor(reader, globalHeap) {
        this._globalHeap = globalHeap;
        this._linkNameOffset = reader.readOffset();
        this._objectHeaderAddress = reader.readOffset();
        this._cacheType = reader.readUint32();
        reader.skipBytes(4); // Reserved
        switch (this._cacheType) {
            case 0:
                reader.skipBytes(16); // Scratch-pad space
                break;
            case 1:
                this._treeAddress = reader.readOffset();
                this._heapAddress = reader.readOffset();
                break;
            default:
                throw new Error('Unsupported cache type \'' + this._cacheType + '\'.');
        }

        this._dataObjectHeader = new H5.DataObjectHeader(reader.move(this._objectHeaderAddress));
    }

    get attributes() {
        if (!this._attributes) {
            this._attributes = {};
            this._dataObjectHeader.messages.forEach((message) => {
                if (message instanceof H5.AttributeMessage) {
                    var name = message.name;
                    var value = message.decodeValue(this._globalHeap);
                    this._attributes[name] = value;
                }
            });
        }
        return this._attributes;
    }
};

H5.DataObjectHeader = class {
    constructor(reader) {
        var version = reader.readByte();
        var messageCount = 0;
        if (version == 1) {
            reader.skipBytes(1);
            messageCount = reader.readUint16();
            this._objectReferenceCount = reader.readUint32();
            this._objectHeaderSize = reader.readUint32();
        }
        else {
            throw new Error('Unsupported data object header version \'' + version + '\'.');
        }

        reader.align(8);

        this._messages = [];
        for (var i = 0; i < messageCount; i++) {
            var messageType = reader.readUint16();
            var dataSize = reader.readUint16();
            var flags = reader.readByte();
            reader.skipBytes(3);
            reader.align(8);
            switch(messageType) {
                case 0x0000: // NIL
                    reader.skipBytes(dataSize);
                    reader.align(8);
                    break;
                case 0x0010: // Object Header Continuation
                    var offset = reader.readOffset();
                    reader = reader.move(offset);
                    break;
                case 0x0011: // Symbol Table Message
                    this._messages.push(new H5.SymbolTableMessage(reader.clone()));
                    reader.skipBytes(dataSize);
                    reader.align(8);
                    break;
                case 0x000C: // Attribute
                    this._messages.push(new H5.AttributeMessage(reader.clone(), flags));
                    reader.skipBytes(dataSize);
                    reader.align(8);
                    break;
                default:
                    throw new Error('Unsupported message type \'' + messageType + '\'.');
            }
        }
    }

    get messages() {
        return this._messages;
    }
};

H5.Message = class {
    constructor(type, data, flags) {
        this._type = type;
        this._data = data;
        this._flags = flags;
    }
};

H5.DataTypeMessage = class {
    constructor(reader) {
        var format = reader.readByte();
        var version = format >> 4;
        this.class = format & 0xf;
        if (version == 1 || version == 2) {
            var flags = reader.readByte() | reader.readByte() << 8 | reader.readByte() << 16;
            this._size = reader.readUint32();

            switch (this.class) {
                case 9: // variable-length
                    this.type = flags & 0x0f;
                    this.padding = (flags >> 4) & 0x0f;
                    this.characterSet = (flags >> 8) & 0x0f;
                    break;
                default:
                    throw new Error('Unknown data type message class \'' + this._class + '\'.');
            }
        }
        else {
            throw new Error('Uknown data type message version \'' + version + '\'.');
        }
    }
};

H5.DataspaceMessage = class {
    constructor(reader) {
        var version = reader.readByte();
        if (version == 1) {
            this.dimensions = reader.readByte();
            this._flags = reader.readByte();
            reader.skipBytes(1);
            reader.skipBytes(4);
        }
        else {
            throw new Error('Unknown dataspace message version \'' + version + '\.');
        }
    }
};

H5.AttributeMessage = class {
    constructor(reader, flags) {
        var version = reader.readByte();
        if (version == 1) {
            reader.skipBytes(1);
            var nameSize = reader.readUint16();
            var dataTypeSize = reader.readUint16();
            var dataspaceSize = reader.readUint16();
            this._name = H5.utf8Decoder.decode(reader.readBytes(nameSize));
            this._name = this._name.replace(/\0/g, '');
            reader.align(8);
            this._dataType = new H5.DataTypeMessage(reader.clone());
            reader.skipBytes(dataTypeSize);
            reader.align(8);
            this._dataspace = new H5.DataspaceMessage(reader.clone());
            reader.skipBytes(dataspaceSize);
            reader.align(8);

            if (this._dataspace.dimensions == 0 &&
                this._dataType.class == 9 &&
                this._dataType.padding == 0 &&
                this._dataType.characterSet == 0) {
                this._length = reader.readUint32();
                this._globalHeapID = new H5.GlobalHeapID(reader);
            }
            else {
                throw new Error('Unsupported attribute message class or type.');
            }
        }
        else {
            throw new Error('Unsupported attribute message version \'' + version + '\'.'); 
        }
    }

    get name() {
        return this._name;
    }

    decodeValue(globalHeap) {
        var globalHeapCollection = globalHeap.get(this._globalHeapID.address);
        if (globalHeapCollection) {
            var globalHeapObject = globalHeapCollection.get(this._globalHeapID.objectIndex);
            if (globalHeapObject != null) {
                return H5.asciiDecoder.decode(globalHeapObject.data);
            }
        }
        return null;
    }
};

H5.SymbolTableMessage = class {
    constructor(reader) {
        this._treeAddress = reader.readOffset();
        this._heapAddress = reader.readOffset();
    }
};

H5.Tree = class {
    constructor(reader) {
        var signature = [ 0x54, 0x52, 0x45, 0x45 ]; // 'TREE'
        while (signature.length > 0) {
            if (reader.readByte() != signature.shift()) { 
                throw new Error("Not a valid 'TREE'.");
            }
        }

        this._type = reader.readByte();
        this._level = reader.readByte();
        this._entriesUsed = reader.readUint16();
        this._leftSiblingAddress = reader.readOffset();
        this._rightSiblingAddress = reader.readOffset();
        // ...
    }
};

H5.Heap = class {
    constructor(reader) {
        var signature = [ 0x48, 0x45, 0x41, 0x50 ]; // 'HEAP'
        while (signature.length > 0) {
            if (reader.readByte() != signature.shift()) { 
                throw new Error("Not a valid 'HEAP'.");
            }
        }

        var version = reader.readByte();
        reader.skipBytes(3);
        this._dataSize = reader.readLength();
        this._offsetToHeadOfFreeList = reader.readLength();
        this._dataAddress = reader.readOffset();
    }
};

H5.GlobalHeap = class {
    constructor(reader) {
        this._collections = {};
        this._reader = reader; 
    }

    get(address) {
        var globalHeapCollection = this._collections[address];
        if (!globalHeapCollection) {
            globalHeapCollection = new H5.GlobalHeapCollection(this._reader.move(address));
            this._collections[address] = globalHeapCollection;
        }
        return globalHeapCollection;
    }
};

H5.GlobalHeapCollection = class {
    constructor(reader) {
        var startPosition = reader.position;
        var signature = [ 0x47, 0x43, 0x4F, 0x4C ]; // 'GCOL'
        while (signature.length > 0) {
            if (reader.readByte() != signature.shift()) { 
                throw new Error("Not a valid 'GCOL'.");
            }
        }
        var version = reader.readByte();
        if (version == 1) {
            reader.skipBytes(3);
            this._objects = {};
            var size = reader.readLength();
            var endPosition = startPosition + size;
            while (reader.position < endPosition) {
                var index = reader.readUint16();
                if (index == 0) {
                    break;
                }
                var heapObject = new H5.GlobalHeapObject(reader);
                this._objects[index] = heapObject;
                reader.align(8);
            }
        }
        else {
            throw new Error('Unknown global heap collection version \'' + version + '\'.');
        }
    }

    get(index) {
        var globalHeapObject = this._objects[index];
        if (globalHeapObject) {
            return globalHeapObject;
        }
        return null;
    }
};

H5.GlobalHeapObject = class {
    constructor(reader) {
        this._referenceCount = reader.readUint16();
        reader.skipBytes(4);
        var size = reader.readLength();
        this._data = reader.readBytes(size);
    }

    get data() {
        return this._data;
    }
};

H5.GlobalHeapID = class {
    constructor(reader) {
        this.address = reader.readOffset();
        this.objectIndex = reader.readUint32(); 
    }
};
