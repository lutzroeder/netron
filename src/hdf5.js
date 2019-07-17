/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental H5/HDF5 JavaScript reader

var hdf5 = hdf5 || {};
var long = long || { Long: require('long') };

hdf5.File = class {

    constructor(buffer) {
        var reader = new hdf5.Reader(buffer, 0);
        this._globalHeap = new hdf5.GlobalHeap(reader);
        if (!reader.match('\x89HDF\r\n\x1A\n')) {
            throw new hdf5.Error('Not a valid HDF5 file.');
        }
        var version = reader.byte();
        switch (version) {
            case 0:
            case 1:
                this._freeSpaceStorageVersion = reader.byte();
                this._rootGroupEntryVersion = reader.byte();
                reader.seek(1);
                this._sharedHeaderMessageVersionFormat = reader.byte();
                reader.initialize();
                reader.seek(1);
                this._groupLeafNodeK = reader.uint16(); // 0x04?
                this._groupInternalNodeK = reader.uint16(); // 0x10?
                reader.seek(4);
                if (version > 0) {
                    this._indexedStorageInternalNodeK = reader.uint16();
                    this.seek(2); // Reserved
                }
                this._baseAddress = reader.offset();
                reader.offset(); // Address of File Free space Info
                this._endOfFileAddress = reader.offset();
                reader.offset(); // Driver Information Block Address
                if (this._baseAddress != 0) {
                    throw new hdf5.Error('Base address is not zero.');
                }
                var rootGroupEntry = new hdf5.SymbolTableEntry(reader);
                this._rootGroup = new hdf5.Group(reader, rootGroupEntry, null, this._globalHeap, '', '');
                break;
            case 2:
            case 3:
                reader.initialize();
                reader.byte();
                this._baseAddress = reader.offset();
                this._superBlockExtensionAddress = reader.offset();
                this._endOfFileAddress = reader.offset();
                var rootGroupObjectHeader = new hdf5.DataObjectHeader(reader.move(reader.offset()));
                this._rootGroup = new hdf5.Group(reader, null, rootGroupObjectHeader, this._globalHeap, '', '');
                break;
            default:
                throw new hdf5.Error('Unsupported Superblock version ' + version + '.');
        }
    }

    get rootGroup() {
        return this._rootGroup;
    }
};

hdf5.Group = class {

    constructor(reader, entry, objectHeader, globalHeap, parentPath, name) {
        this._reader = reader;
        this._entry = entry;
        this._dataObjectHeader = objectHeader;
        this._globalHeap = globalHeap;
        this._name = name;
        this._path = parentPath == '/' ? (parentPath + name) : (parentPath + '/' + name);
    }

    get name() {
        return this._name;
    }

    get path() {
        return this._path;
    }

    group(path) {
        this.decodeGroups();
        var index = path.indexOf('/');
        if (index != -1) {
            var childPath = path.substring(index + 1);
            var subPath = path.substring(0, index);
            var subGroup = this.group(subPath);
            if (subGroup != null) {
                return subGroup.group(childPath);
            }
        }
        else {
            var group = this._groupMap[path];
            if (group) {
                return group;
            }
        }
        return null;
    }

    get groups() {
        this.decodeGroups();
        return this._groups;
    }

    attribute(name) {
        this.decodeDataObject();
        return this._attributes[name];
    }

    get attributes() {
        this.decodeDataObject();
        return this._attributes;
    }

    get value() { 
        this.decodeDataObject();
        return this._value;
    }

    decodeDataObject() {
        if (!this._dataObjectHeader) {
            this._dataObjectHeader = new hdf5.DataObjectHeader(this._reader.move(this._entry.objectHeaderAddress));
        }
        if (!this._attributes) {
            this._attributes = {};
            for (var attribute of this._dataObjectHeader.attributes) {
                var name = attribute.name;
                var value = attribute.decodeValue(this._globalHeap);
                this._attributes[name] = value;
            }
            this._value = null;
            var datatype = this._dataObjectHeader.datatype;
            var dataspace = this._dataObjectHeader.dataspace;
            var dataLayout = this._dataObjectHeader.dataLayout;
            if (datatype && dataspace && dataLayout) {
                this._value = new hdf5.Variable(this._reader, this._globalHeap, datatype, dataspace, dataLayout);
            }
        }
    }

    decodeGroups() {
        if (!this._groups) {
            this._groupMap = {};
            this._groups = [];
            if (this._entry) {
                if (this._entry.treeAddress || this._entry.heapAddress) {
                    var heap = new hdf5.Heap(this._reader.move(this._entry.heapAddress));
                    var tree = new hdf5.Tree(this._reader.move(this._entry.treeAddress));
                    for (var node of tree.nodes) {
                        for (var entry of node.entries) {
                            var name = heap.getString(entry.linkNameOffset);
                            var group = new hdf5.Group(this._reader, entry, null, this._globalHeap, this._path, name);
                            this._groups.push(group);
                            this._groupMap[name] = group;
                        }
                    }
                }    
            }
            else {
                this.decodeDataObject();
                for (var link of this._dataObjectHeader.links) {
                    if (Object.prototype.hasOwnProperty.call(link, 'objectHeaderAddress')) {
                        var objectHeader = new hdf5.DataObjectHeader(this._reader.move(link.objectHeaderAddress));
                        var linkGroup = new hdf5.Group(this._reader, null, objectHeader, this._globalHeap, this._path, link.name);
                        this._groups.push(linkGroup);
                        this._groupMap[name] = linkGroup;
                    }
                }
            }
        }
    }
};

hdf5.Variable = class {
    constructor(reader, globalHeap, datatype, dataspace, dataLayout) {
        this._reader = reader;
        this._globalHeap = globalHeap;
        this._datatype = datatype;
        this._dataspace = dataspace;
        this._dataLayout = dataLayout;
    }

    get type () {
        return this._datatype.type;
    }

    get shape() {
        return this._dataspace.shape;
    }

    get value() {
        if (this._dataLayout.address) {
            var reader = this._reader.move(this._dataLayout.address);
            var data = this._dataspace.read(this._datatype, reader);
            var value = this._dataspace.decode(this._datatype, data, data, this._globalHeap);
            return value;
        }
        return null;
    }

    get rawData() {
        if (this._dataLayout.address) {
            var reader = this._reader.move(this._dataLayout.address);
            return reader.bytes(this._dataLayout.size);
        }
        return null;
    }
};

hdf5.Reader = class {
    constructor(buffer) {
        if (buffer) {
            this._buffer = buffer;
            this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
            this._position = 0;
            this._offset = 0;
        }
    }

    initialize() {
        this._offsetSize = this.byte();
        this._lengthSize = this.byte();
    }

    int8() {
        var value = this._dataView.getInt8(this._position + this._offset);
        this._offset++;
        return value;
    }

    byte() {
        var value = this._dataView.getUint8(this._position + this._offset);
        this._offset++;
        return value;
    }

    seek(offset) {
        this._offset += offset;
    }

    bytes(length) {
        var data = this._buffer.subarray(this._position + this._offset, this._position + this._offset + length);
        this._offset += length;
        return data;
    }

    int16() {
        var value = this._dataView.getInt16(this._position + this._offset, true);
        this._offset += 2;
        return value;
    }

    uint16() {
        var value = this._dataView.getUint16(this._position + this._offset, true);
        this._offset += 2;
        return value;
    }

    int32() {
        var value = this._dataView.getInt32(this._position + this._offset, true);
        this._offset += 4;
        return value;
    }

    uint32() {
        var value = this._dataView.getUint32(this._position + this._offset, true);
        this._offset += 4;
        return value;
    }

    int64() {
        var lo = this._dataView.getUint32(this._position + this._offset, true);
        var hi = this._dataView.getUint32(this._position + this._offset + 4, true);
        this._offset += 8;
        return new long.Long(lo, hi, false).toNumber();
    }

    uint64() {
        var lo = this._dataView.getUint32(this._position + this._offset, true);
        var hi = this._dataView.getUint32(this._position + this._offset + 4, true);
        this._offset += 8;
        return new long.Long(lo, hi, true).toNumber();
    }

    uint(type) {
        switch (type) {
            case 0: return this.byte();
            case 1: return this.uint16();
            case 2: return this.uint32();
            case 3: return this.uint64();
        }
    }

    float16() {
        var value = this._dataView.getUint16(this._position + this._offset, true);
        this._offset += 2;
        // decode float16 value
        var s = (value & 0x8000) >> 15;
        var e = (value & 0x7C00) >> 10;
        var f = value & 0x03FF;
        if(e == 0) {
            return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
        }
        else if (e == 0x1F) {
            return f ? NaN : ((s ? -1 : 1) * Infinity);
        }
        return (s ? -1 : 1) * Math.pow(2, e-15) * (1 + (f / Math.pow(2, 10)));
    }

    float32() {
        var value = this._dataView.getFloat32(this._position + this._offset, true);
        this._offset += 4;
        return value;
    }

    float64() {
        var value = this._dataView.getFloat64(this._position + this._offset, true);
        this._offset += 8;
        return value;
    }

    string(size, encoding) {
        if (!size || size == -1) {
            var position = this._position + this._offset;
            while (this._buffer[position] != 0) {
                position++;
            }
            size = position - this._position + this._offset + 1;
        }
        var data = this.bytes(size);
        return hdf5.Reader.decode(data, encoding);
    }

    static decode(data, encoding) {
        var text = '';
        if (encoding == 'utf-8') {
            if (!hdf5.Reader._utf8Decoder) {
                hdf5.Reader._utf8Decoder = new TextDecoder('utf-8');
            }
            text = hdf5.Reader._utf8Decoder.decode(data);
        }
        else {
            if (!hdf5.Reader._asciiDecoder) {
                hdf5.Reader._asciiDecoder = new TextDecoder('ascii');
            }
            text = hdf5.Reader._asciiDecoder.decode(data);
        }
        return text.replace(/\0/g, '');
    }

    offset() { 
        switch (this._offsetSize) {
            case 8:
                return this.uint64();
            case 4:
                return this.uint32(); 
        }
        throw new hdf5.Error('Unsupported offset size \'' + this._offsetSize + '\'.');
    }

    length() {
        switch (this._lengthSize) {
            case 8:
                return this.uint64();
            case 4:
                return this.uint32(); 
        }
        throw new hdf5.Error('Unsupported length size \'' + this._lengthSize + '\'.');
    }

    move(position) {
        var reader = new hdf5.Reader(null);
        reader._buffer = this._buffer;
        reader._dataView = this._dataView;
        reader._position = position;
        reader._offset = 0;
        reader._offsetSize = this._offsetSize;
        reader._lengthSize = this._lengthSize;
        return reader;
    }

    clone() {
        var reader =  new hdf5.Reader(this._buffer, this._position);
        reader._buffer = this._buffer;
        reader._dataView = this._dataView;
        reader._position = this._position;
        reader._offset = this._offset;
        reader._offsetSize = this._offsetSize;
        reader._lengthSize = this._lengthSize;
        return reader;
    }

    align(mod) {
        if (this._offset % mod != 0) {
            this._offset = (Math.floor(this._offset / mod) + 1) * mod;
        }
    }

    match(text) {
        if (this._position + this._offset + text.length > this._buffer.length) {
            return false;
        }
        var offset = this._offset;
        var buffer = this.bytes(text.length);
        for (var i = 0; i < text.length; i++) {
            if (text.charCodeAt(i) != buffer[i]) {
                this._offset = offset;
                return false;
            }
        }
        return true;
    }

    get position() {
        return this._position + this._offset;
    }

    get size() {
        return this._buffer.length;
    }
};

hdf5.SymbolTableNode = class {
    constructor(reader) {
        if (!reader.match('SNOD')) {
            throw new hdf5.Error("Not a valid 'SNOD' block.");
        }
        var version = reader.byte();
        if (version == 1) {
            reader.seek(1);
            var entriesUsed = reader.uint16();
            this.entries = [];
            for (var i = 0; i < entriesUsed; i++) {
                this.entries.push(new hdf5.SymbolTableEntry(reader));
            }
        }
        else {
            throw new hdf5.Error('Unsupported symbol table node version \'' + version + '\'.');
        }        
    }
};

hdf5.SymbolTableEntry = class {
    constructor(reader) {
        this.linkNameOffset = reader.offset();
        this.objectHeaderAddress = reader.offset();
        var cacheType = reader.uint32();
        reader.seek(4); // Reserved
        switch (cacheType) {
            case 0:
                break;
            case 1:
                var scratchReader = reader.clone();
                this.treeAddress = scratchReader.offset();
                this.heapAddress = scratchReader.offset();
                break;
            default:
                throw new hdf5.Error('Unsupported cache type \'' + cacheType + '\'.');
        }
        reader.seek(16); // Scratch-pad space
    }
};

hdf5.DataObjectHeader = class {

    constructor(reader) {
        this.attributes = [];
        this.links = [];
        this.continuations = [];
        var version = reader.match('OHDR') ? reader.byte() : reader.byte();
        switch (version) {
            case 1:
                this.constructor_v1(reader);
                break;
            case 2:
                this.constructor_v2(reader);
                break;
            default:
                throw new hdf5.Error('Unsupported data object header version \'' + version + '\'.');
        }
    }

    constructor_v1(reader) {
        reader.seek(1);
        var messageCount = reader.uint16();
        reader.uint32();
        var objectHeaderSize = reader.uint32();
        reader.align(8);
        var end = reader.position + objectHeaderSize;
        for (var i = 0; i < messageCount; i++) {
            var messageType = reader.uint16();
            var messageSize = reader.uint16();
            var messageFlags = reader.byte();
            reader.seek(3);
            reader.align(8);
            var next = this.readMessage(reader, messageType, messageSize, messageFlags);
            if ((!next || reader.position >= end) && this.continuations.length > 0) {
                var continuation = this.continuations.shift();
                reader = reader.move(continuation.offset);
                end = continuation.offset + continuation.length;
            }
            else {
                reader.align(8);
            }
        }
    } 

    constructor_v2(reader) {
        var flags = reader.byte();
        if ((flags & 0x20) != 0) {
            reader.uint32();
            reader.uint32();
            reader.uint32();
            reader.uint32();
        }
        if ((flags & 0x10) != 0) {
            reader.uint16();
            reader.uint16();
        }
        var size = reader.uint(flags & 0x03);
        var next = true;
        var end = reader.position + size;
        while (next && reader.position < end) {
            var messageType = reader.byte();
            var messageSize = reader.uint16();
            var messageFlags = reader.byte();
            if (reader.position < end) {
                if ((flags & 0x04) != 0) {
                    reader.uint16();
                }
                next = this.readMessage(reader, messageType, messageSize, messageFlags);
            } 
            if ((!next || reader.position >= end) && this.continuations.length > 0) {
                var continuation = this.continuations.shift();
                reader = reader.move(continuation.offset);
                end = continuation.offset + continuation.length;
                if (!reader.match('OCHK')) {
                    throw new hdf5.Error("Invalid continuation block signature.");
                }
                next = true;
            }
        }
    }

    readMessage(reader, type, size, flags) {
        switch(type) {
            case 0x0000: // NIL
                return false;
            case 0x0001: // Dataspace
                this.dataspace = (size != 4 || flags != 1) ? new hdf5.Dataspace(reader.clone()) : null;
                break;
            case 0x0002: // Link Info
                this.linkInfo = new hdf5.LinkInfo(reader.clone());
                break;
            case 0x0003: // Datatype
                this.datatype = new hdf5.Datatype(reader.clone());
                break;
            case 0x0005: // Fill Value
                this.fillValue = new hdf5.FillValue(reader.clone());
                break;
            case 0x0006: // Link
                this.links.push(new hdf5.Link(reader.clone()));
                break;
            case 0x0008: // Data Layout
                this.dataLayout = new hdf5.DataLayout(reader.clone()); 
                break;
            case 0x000A: // Group Info
                this.groupInfo = new hdf5.GroupInfo(reader.clone());
                break;
            case 0x000C: // Attribute
                this.attributes.push(new hdf5.Attribute(reader.clone()));
                break;
            case 0x000D: // Object Comment Message
                this.comment = reader.string(-1, 'ascii');
                break; 
            case 0x0010: // Object Header Continuation
                this.continuations.push(new hdf5.ObjectHeaderContinuation(reader.clone()));
                break;
            case 0x0011: // Symbol Table
                this.symbolTable = new hdf5.SymbolTable(reader.clone());
                break;
            case 0x0012: // Object Modification Time
                reader.seek(size);
                reader.align(8);
                break;
            case 0x0015: // Attribute Info
                this.attributeInfo = new hdf5.AttributeInfo(reader.clone());
                break;
            default:
                throw new hdf5.Error('Unsupported message type \'' + type + '\'.');
        }
        reader.seek(size);
        return true;
    }
};

hdf5.Message = class {

    constructor(type, data, flags) {
        this._type = type;
        this._data = data;
        this._flags = flags;
    }
};

hdf5.Datatype = class {

    constructor(reader) {
        var format = reader.byte();
        var version = format >> 4;
        this._class = format & 0xf;
        switch (version) {
            case 1:
            case 2:
                this._flags = reader.byte() | reader.byte() << 8 | reader.byte() << 16;
                this._size = reader.uint32();
                break;
            default:
                throw new hdf5.Error('Unsupported datatype version \'' + version + '\'.');
        }
    }

    get type() {
        switch (this._class) {
            case 0: // fixed-point
                throw new hdf5.Error("Unsupported datatype (class=0, size=" + this._size + ", flags=" + this._flags + ")."); 
            case 1: // floating-point
                if (this._size == 2 && this._flags == 0x0f20) {
                    return 'float16';
                }
                else if (this._size == 4 && this._flags == 0x1f20) {
                    return 'float32';
                }
                else if (this._size == 8 && this._flags == 0x3f20) {
                    return 'float64';
                }
                break;
            case 3: // string
                return 'string';
            case 5: // opaque
                return 'uint8[]';
            case 9: // variable-length
                if ((this._flags & 0x0f) == 1) { // type
                    return 'char[]';
                }
                break;
        }
        throw new hdf5.Error('Unsupported datatype class \'' + this._class + '\'.');
    }

    read(reader) {
        switch (this._class) {
            case 0: // fixed-point
                if (this._size == 1) {
                    return ((this._flags & 0x8) != 0) ? reader.int8() : reader.byte();
                }
                else if (this._size == 2) {
                    return ((this._flags & 0x8) != 0) ? reader.int16() : reader.uint16();
                }
                else if (this._size == 4) {
                    return ((this._flags & 0x8) != 0) ? reader.int32() : reader.uint32();
                }
                else if (this._size == 8) {
                    return ((this._flags & 0x8) != 0) ? reader.int64() : reader.uint64();
                }
                throw new hdf5.Error('Unsupported fixed-point datatype.');
            case 1: // floating-point
                if (this._size == 2 && this._flags == 0x0f20) {
                    return reader.float16();
                }
                else if (this._size == 4 && this._flags == 0x1f20) {
                    return reader.float32();
                }
                else if (this._size == 8 && this._flags == 0x3f20) {
                    return reader.float64();
                }
                throw new hdf5.Error('Unsupported floating-point datatype.');
            case 3: // string
                switch ((this._flags >> 8) & 0x0f) { // character set
                    case 0:
                        return hdf5.Reader.decode(reader.bytes(this._size), 'ascii');
                    case 1:
                        return hdf5.Reader.decode(reader.bytes(this._size), 'utf-8');
                }
                throw new hdf5.Error('Unsupported character encoding.');
            case 5: // opaque
                return reader.bytes(this._size);
            case 9: // variable-length
                return {
                    length: reader.uint32(),
                    globalHeapID: new hdf5.GlobalHeapID(reader)
                };
        }
        throw new hdf5.Error('Unsupported datatype class \'' + this._class + '\'.');
    }

    decode(data, globalHeap) {
        switch (this._class) {
            case 0: // fixed-point
                return data;
            case 1: // floating-point
                return data;
            case 3: // string
                return data;
            case 5: // opaque
                return data;
            case 9: // variable-length
                var globalHeapObject = globalHeap.get(data.globalHeapID);
                if (globalHeapObject != null) {
                    var characterSet = (this._flags >> 8) & 0x0f; 
                    switch (characterSet) {
                        case 0:
                            return hdf5.Reader.decode(globalHeapObject.data, 'ascii');
                        case 1:
                            return hdf5.Reader.decode(globalHeapObject.data, 'utf-8');
                    }
                    throw new hdf5.Error('Unsupported character encoding.');
                }
                break;
            default:
                throw new hdf5.Error('Unsupported datatype class \'' + this._class + '\'.');
        }
        return null;
    }
};

hdf5.Dataspace = class {

    constructor(reader) {
        this._sizes = [];
        var version = reader.byte();
        switch (version) {
            case 1:
                this._dimensions = reader.byte();
                this._flags = reader.byte();
                reader.seek(1);
                reader.seek(4);
                for (var i = 0; i < this._dimensions; i++) {
                    this._sizes.push(reader.length());
                }
                if ((this._flags & 0x01) != 0) {
                    this._maxSizes = [];
                    for (var j = 0; j < this._dimensions; j++) {
                        this._maxSizes.push(reader.length());
                        if (this._maxSizes[j] != this._sizes[j]) {
                            throw new hdf5.Error('Max size is not supported.');
                        }
                    }
                }
                if ((this._flags & 0x02) != 0) {
                    throw new hdf5.Error('Permutation indices not supported.');
                }
                break;
            case 2:
                this._dimensions = reader.byte();
                this._flags = reader.byte();
                this._type = reader.byte(); // 0 scalar, 1 simple, 2 null
                for (var k = 0; k < this._dimensions; k++) {
                    this._sizes.push(reader.length());
                }
                if ((this._flags & 0x01) != 0) {
                    this._maxSizes = [];
                    for (var l = 0; l < this._dimensions; l++) {
                        this._maxSizes.push(reader.length());
                    }
                }
                break;
            default:
                throw new hdf5.Error("Unsupported dataspace message version '" + version + "'.");
    
        }
    }

    get shape() {
        return this._sizes;
    }

    read(datatype, reader) {
        if (this._dimensions == 0) {
            return datatype.read(reader);
        }
        return this.readArray(datatype, reader, this._sizes, 0);
    }

    readArray(datatype, reader, shape, dimension) {
        var array = [];
        var size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (var i = 0; i < size; i++) {
                array.push(datatype.read(reader));
            }
        }
        else {
            for (var j = 0; j < size; j++) {
                array.push(this.readArray(datatype, reader, shape, dimension + 1));
            }
        }     
        return array;
    }

    decode(datatype, data, globalHeap) {
        if (this._dimensions == 0) {
            return datatype.decode(data, globalHeap);
        }
        return this.decodeArray(datatype, data, globalHeap, this._sizes, 0);
    }

    decodeArray(datatype, data, globalHeap, shape, dimension) {
        var size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (var i = 0; i < size; i++) {
                data[i] = datatype.decode(data[i], globalHeap);
            }
        }
        else {
            for (var j = 0; j < size; j++) {
                data[j] = this.decodeArray(datatype, data[j], shape, dimension + 1);
            }
        }
        return data;
    }
};

hdf5.LinkInfo = class {

    constructor(reader) {
        var version = reader.byte();
        switch (version) {
            case 0:
                var flags = reader.byte();
                if ((flags & 1) != 0) {
                    this.maxCreationIndex = reader.uint64();
                }
                this.fractalHeapAddress = reader.offset();
                this.nameIndexTreeAddress = reader.offset();
                if ((flags & 2) != 0) {
                    this.creationOrderIndexTreeAddress = reader.offset();
                }
                break;
            default:
                throw new hdf5.Error("Unsupported link info message version '" + version + "'.");
        }
    }
};

hdf5.FillValue = class {

    constructor(reader) {
        var version = reader.byte();
        switch (version) {
            case 2:
                reader.byte();
                reader.byte();
                var valueDefined = reader.byte();
                if (valueDefined == 1) {
                    var size = reader.uint32();
                    this.data = reader.bytes(size);
                }
                break;
            case 3:
                // var flags = reader.byte();
                // if ((flags & 0x20) != 0) {
                // }                
                break;
            default:
                throw new hdf5.Error('Unsupported fill value version \'' + version + '\'.');
        }
    }
};

hdf5.Link = class {

    constructor(reader) {
        var version = reader.byte();
        switch (version) {
            case 1:
                var flags = reader.byte();
                this.type = (flags & 0x08) != 0 ? reader.byte() : 0;
                if ((flags & 0x04) != 0) {
                    this.creationOrder = reader.uint32();
                }
                var linkNameEncoding = ((flags & 0x10) != 0 && reader.byte() == 1) ? 'utf-8' : 'ascii';
                this.name = reader.string(reader.uint(flags & 0x03), linkNameEncoding);
                switch (this.type) {
                    case 0: // hard link
                        this.objectHeaderAddress = reader.offset();
                        break;
                    case 1: // soft link
                        break;
                }
                break;
            default:
                throw new hdf5.Error('Unsupported link message version \'' + version + '\'.');
        }
    }
};

hdf5.DataLayout = class {

    constructor(reader) {
        var version = reader.byte();
        switch (version) {
            case 3:
                var layoutClass = reader.byte();
                switch (layoutClass) {
                    case 1: // Contiguous Storage
                        this.address = reader.offset();
                        this.size = reader.length();
                        break;
                    default:
                        throw new hdf5.Error('Unsupported data layout class \'' + layoutClass + '\'.');
                }
                break;
            default:
                throw new hdf5.Error('Unsupported data layout version \'' + version + '\'.');
        }
    }
};

hdf5.GroupInfo = class {

    constructor(reader) {
        var version = reader.byte();
        switch (version) {
            case 0:
                var flags = reader.byte();
                if ((flags & 0x01) != 0) {
                    this.maxCompactLinks = reader.uint16();
                    this.minDenseLinks = reader.uint16();
                }
                if ((flags & 0x02) != 0) {
                    this.estimatedEntriesNumber = reader.uint16();
                    this.estimatedLinkNameLengthEntires = reader.uint16();
                }
                break;
            default:
                throw new hdf5.Error('Unsupported group info version \'' + version + '\'.');
        }
    }
};

hdf5.Attribute = class {

    constructor(reader) {
        var version = reader.byte();
        switch (version) {
            case 1:
                this.constructor_v1(reader);
                break;
            case 3:
                this.constructor_v3(reader);
                break;
            default:
                throw new hdf5.Error('Unsupported attribute message version \'' + version + '\'.'); 
        }
    }

    constructor_v1(reader) {
        reader.seek(1);
        var nameSize = reader.uint16();
        var datatypeSize = reader.uint16();
        var dataspaceSize = reader.uint16();
        this.name = reader.string(nameSize, 'utf-8');
        reader.align(8);
        this._datatype = new hdf5.Datatype(reader.clone());
        reader.seek(datatypeSize);
        reader.align(8);
        this._dataspace = new hdf5.Dataspace(reader.clone());
        reader.seek(dataspaceSize);
        reader.align(8);
        this._data = this._dataspace.read(this._datatype, reader);
    }

    constructor_v3(reader) {
        reader.byte();
        var nameSize = reader.uint16();
        var datatypeSize = reader.uint16();
        var dataspaceSize = reader.uint16();
        var encoding = reader.byte() == 1 ? 'utf-8' : 'ascii';
        this.name = reader.string(nameSize, encoding);
        this._datatype = new hdf5.Datatype(reader.clone());
        reader.seek(datatypeSize);
        this._dataspace = new hdf5.Dataspace(reader.clone());
        reader.seek(dataspaceSize);
        this._data = this._dataspace.read(this._datatype, reader);
    }

    decodeValue(globalHeap) {
        if (this._data) {
            return this._dataspace.decode(this._datatype, this._data, globalHeap);
        }
        return null;
    }
};

hdf5.ObjectHeaderContinuation = class {

    constructor(reader) {
        this.offset = reader.offset();
        this.length = reader.length();
    }
};

hdf5.SymbolTable = class {

    constructor(reader) {
        this._treeAddress = reader.offset();
        this._heapAddress = reader.offset();
    }
};

hdf5.AttributeInfo = class {

    constructor(reader) {
        var version = reader.byte();
        switch (version) {
            case 0:
                var flags = reader.byte();
                if ((flags & 1) != 0) {
                    this.maxCreationIndex = reader.uint64();
                }
                this.fractalHeapAddress = reader.offset();
                this.attributeNameTreeAddress = reader.offset();
                if ((flags & 2) != 0) {
                    this.attributeCreationOrderTreeAddress = reader.offset();
                }
                break;
            default:
                throw new hdf5.Error('Unsupported attribute info message version \'' + version + '\'.'); 
        }
    }
};

hdf5.Tree = class {

    constructor(reader) {
        if (!reader.match('TREE')) {
            throw new hdf5.Error("Not a valid 'TREE' block.");
        }
        var type = reader.byte();
        var level = reader.byte();
        var entriesUsed = reader.uint16();
        reader.offset();
        reader.offset();
        this.nodes = [];
        if (type == 0) {
            for (var i = 0; i < entriesUsed; i++) {
                reader.length();
                var childPointer = reader.offset();
                if (level == 0) {
                    this.nodes.push(new hdf5.SymbolTableNode(reader.move(childPointer)));
                }
                else {
                    var tree = new hdf5.Tree(reader.move(childPointer));
                    this.nodes = this.nodes.concat(tree.nodes);
                }
            }
        }
        else {
            throw new hdf5.Error('Unsupported B-Tree node type \'' + type + '\'.');
        }
    }
};

hdf5.Heap = class {

    constructor(reader) {
        this._reader = reader;
        if (!reader.match('HEAP')) {
            throw new hdf5.Error("Not a valid 'HEAP' block.");
        }
        var version = reader.byte();
        if (version == 0) {
            reader.seek(3);
            this._dataSize = reader.length();
            this._offsetToHeadOfFreeList = reader.length();
            this._dataAddress = reader.offset();
        }
        else {
            throw new hdf5.Error('Unsupported Local Heap version \'' + version + '\'.');

        }
    }

    getString(offset) {
        var reader = this._reader.move(this._dataAddress + offset);
        return reader.string(-1, 'utf-8');
    }
};

hdf5.GlobalHeap = class {
    constructor(reader) {
        this._reader = reader; 
        this._collections = {};
    }

    getCollection(address) {
        var globalHeapCollection = this._collections[address];
        if (!globalHeapCollection) {
            globalHeapCollection = new hdf5.GlobalHeapCollection(this._reader.move(address));
            this._collections[address] = globalHeapCollection;
        }
        return globalHeapCollection;
    }

    get(globalHeapID) {
        var globalHeapObject = null;
        var globalHeapCollection = this.getCollection(globalHeapID.address);
        if (globalHeapCollection != null) {
            globalHeapObject = globalHeapCollection.getObject(globalHeapID.objectIndex);
        }
        return globalHeapObject;
    }
};

hdf5.GlobalHeapCollection = class {
    constructor(reader) {
        var startPosition = reader.position;
        if (!reader.match('GCOL')) {
            throw new hdf5.Error("Not a valid 'GCOL' block.");
        }
        var version = reader.byte();
        if (version == 1) {
            reader.seek(3);
            this._objects = {};
            var size = reader.length();
            var endPosition = startPosition + size;
            while (reader.position < endPosition) {
                var index = reader.uint16();
                if (index == 0) {
                    break;
                }
                var heapObject = new hdf5.GlobalHeapObject(reader);
                this._objects[index] = heapObject;
                reader.align(8);
            }
        }
        else {
            throw new hdf5.Error('Unsupported global heap collection version \'' + version + '\'.');
        }
    }

    getObject(objectIndex) {
        var globalHeapObject = this._objects[objectIndex];
        if (globalHeapObject) {
            return globalHeapObject;
        }
        return null;
    }
};

hdf5.GlobalHeapObject = class {

    constructor(reader) {
        reader.uint16();
        reader.seek(4);
        var size = reader.length();
        this.data = reader.bytes(size);
    }
};

hdf5.GlobalHeapID = class {

    constructor(reader) {
        this.address = reader.offset();
        this.objectIndex = reader.uint32(); 
    }
};

hdf5.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'HDF5 Error';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.File = hdf5.File; 
}