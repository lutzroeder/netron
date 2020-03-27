/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental HDF5 JavaScript reader

var hdf5 = hdf5 || {};
var long = long || { Long: require('long') };
var zip = zip || require('./zip');

hdf5.File = class {

    constructor(buffer) {
        // https://support.hdfgroup.org/HDF5/doc/H5.format.html
        const reader = new hdf5.Reader(buffer, 0);
        this._globalHeap = new hdf5.GlobalHeap(reader);
        if (!reader.match('\x89HDF\r\n\x1A\n')) {
            throw new hdf5.Error('Not a valid HDF5 file.');
        }
        const version = reader.byte();
        switch (version) {
            case 0:
            case 1: {
                this._freeSpaceStorageVersion = reader.byte();
                this._rootGroupEntryVersion = reader.byte();
                reader.skip(1);
                this._sharedHeaderMessageVersionFormat = reader.byte();
                reader.initialize();
                reader.skip(1);
                this._groupLeafNodeK = reader.uint16(); // 0x04?
                this._groupInternalNodeK = reader.uint16(); // 0x10?
                reader.skip(4);
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
                const rootGroupEntry = new hdf5.SymbolTableEntry(reader);
                this._rootGroup = new hdf5.Group(reader, rootGroupEntry, null, this._globalHeap, '', '');
                break;
            }
            case 2:
            case 3: {
                reader.initialize();
                reader.byte();
                this._baseAddress = reader.offset();
                this._superBlockExtensionAddress = reader.offset();
                this._endOfFileAddress = reader.offset();
                const rootGroupObjectHeader = new hdf5.DataObjectHeader(reader.at(reader.offset()));
                this._rootGroup = new hdf5.Group(reader, null, rootGroupObjectHeader, this._globalHeap, '', '');
                break;
            }
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
        this._decodeGroups();
        const index = path.indexOf('/');
        if (index != -1) {
            const childPath = path.substring(index + 1);
            const subPath = path.substring(0, index);
            const subGroup = this.group(subPath);
            if (subGroup != null) {
                return subGroup.group(childPath);
            }
        }
        else {
            const group = this._groupMap[path];
            if (group) {
                return group;
            }
        }
        return null;
    }

    get groups() {
        this._decodeGroups();
        return this._groups;
    }

    attribute(name) {
        this._decodeDataObject();
        return this._attributes[name];
    }

    get attributes() {
        this._decodeDataObject();
        return this._attributes;
    }

    get value() { 
        this._decodeDataObject();
        return this._value;
    }

    _decodeDataObject() {
        if (!this._dataObjectHeader) {
            this._dataObjectHeader = new hdf5.DataObjectHeader(this._reader.at(this._entry.objectHeaderAddress));
        }
        if (!this._attributes) {
            this._attributes = {};
            for (const attribute of this._dataObjectHeader.attributes) {
                const name = attribute.name;
                const value = attribute.decodeValue(this._globalHeap);
                this._attributes[name] = value;
            }
            this._value = null;
            const datatype = this._dataObjectHeader.datatype;
            const dataspace = this._dataObjectHeader.dataspace;
            const dataLayout = this._dataObjectHeader.dataLayout;
            const filterPipeline = this._dataObjectHeader.filterPipeline;
            if (datatype && dataspace && dataLayout) {
                this._value = new hdf5.Variable(this._reader, this._globalHeap, datatype, dataspace, dataLayout, filterPipeline);
            }
        }
    }

    _decodeGroups() {
        if (!this._groups) {
            this._groupMap = {};
            this._groups = [];
            if (this._entry) {
                if (this._entry.treeAddress || this._entry.heapAddress) {
                    const heap = new hdf5.Heap(this._reader.at(this._entry.heapAddress));
                    const tree = new hdf5.Tree(this._reader.at(this._entry.treeAddress));
                    for (const node of tree.nodes) {
                        for (const entry of node.entries) {
                            const name = heap.getString(entry.linkNameOffset);
                            const group = new hdf5.Group(this._reader, entry, null, this._globalHeap, this._path, name);
                            this._groups.push(group);
                            this._groupMap[name] = group;
                        }
                    }
                }    
            }
            else {
                this._decodeDataObject();
                for (const link of this._dataObjectHeader.links) {
                    if (Object.prototype.hasOwnProperty.call(link, 'objectHeaderAddress')) {
                        const name = link.name;
                        const objectHeader = new hdf5.DataObjectHeader(this._reader.at(link.objectHeaderAddress));
                        const linkGroup = new hdf5.Group(this._reader, null, objectHeader, this._globalHeap, this._path, name);
                        this._groups.push(linkGroup);
                        this._groupMap[name] = linkGroup;
                    }
                }
            }
        }
    }
};

hdf5.Variable = class {

    constructor(reader, globalHeap, datatype, dataspace, dataLayout, filterPipeline) {
        this._reader = reader;
        this._globalHeap = globalHeap;
        this._datatype = datatype;
        this._dataspace = dataspace;
        this._dataLayout = dataLayout;
        this._filterPipeline = filterPipeline;
    }

    get type () {
        return this._datatype.type;
    }

    get littleEndian() {
        return this._datatype.littleEndian;
    }

    get shape() {
        return this._dataspace.shape;
    }

    get value() {
        const data = this.data;
        if (data) {
            const reader = new hdf5.Reader(data);
            const array = this._dataspace.read(this._datatype, reader);
            return this._dataspace.decode(this._datatype, array, array, this._globalHeap);
        }
        return null;
    }

    get data() {
        switch (this._dataLayout.layoutClass) {
            case 1: // Contiguous
                if (this._dataLayout.address) {
                    return this._reader.at(this._dataLayout.address).bytes(this._dataLayout.size);
                }
                break;
            case 2: { // Chunked
                const tree = new hdf5.Tree(this._reader.at(this._dataLayout.address), this._dataLayout.dimensionality);
                if (this._dataLayout.dimensionality == 2 && this._dataspace.shape.length == 1) {
                    let size = this._dataLayout.datasetElementSize;
                    for (let i = 0; i < this._dataspace.shape.length; i++) {
                        size *= this._dataspace.shape[i];
                    }
                    const data = new Uint8Array(size);
                    for (const node of tree.nodes) {
                        if (node.fields.length !== 2 || node.fields[1] !== 0) {
                            return null;
                        }
                        if (node.filterMask !== 0) {
                            return null;
                        }
                        const start = node.fields[0] * this._dataLayout.datasetElementSize;
                        let chunk = node.data;
                        if (this._filterPipeline) {
                            for (const filter of this._filterPipeline.filters) {
                                chunk = filter.decode(chunk);
                            }
                        }
                        for (let i = 0; i < chunk.length; i++) {
                            data[start + i] = chunk[i];
                        }
                    }
                    return data;
                }
                break;
            }
            default: {
                throw new hdf5.Error("Unknown data layout class '" + this.layoutClass + "'.");
            }
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

    skip(offset) {
        this._offset += offset;
        if (this._position + this._offset > this._buffer.length) {
            throw new hdf5.Error('Expected ' + (this._position + this._offset - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    int8() {
        const offset = this._offset;
        this.skip(1);
        return this._dataView.getInt8(this._position + offset);
    }

    byte() {
        const offset = this._offset;
        this.skip(1);
        return this._dataView.getUint8(this._position + offset);
    }

    bytes(length) {
        const offset = this._offset;
        this.skip(length);
        return this._buffer.subarray(this._position + offset, this._position + this._offset);
    }

    int16() {
        const offset = this._offset;
        this.skip(2);
        return this._dataView.getInt16(this._position + offset, true);
    }

    uint16() {
        const offset = this._offset;
        this.skip(2);
        return this._dataView.getUint16(this._position + offset, true);
    }

    int32() {
        const offset = this._offset;
        this.skip(4);
        return this._dataView.getInt32(this._position + offset, true);
    }

    uint32() {
        const offset = this._offset;
        this.skip(4);
        return this._dataView.getUint32(this._position + offset, true);
    }

    int64() {
        const offset = this._offset;
        this.skip(8);
        const lo = this._dataView.getUint32(this._position + offset, true);
        const hi = this._dataView.getUint32(this._position + offset + 4, true);
        return new long.Long(lo, hi, false).toNumber();
    }

    uint64() {
        const offset = this._offset;
        this.skip(8);
        const lo = this._dataView.getUint32(this._position + offset, true);
        const hi = this._dataView.getUint32(this._position + offset + 4, true);
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
        const offset = this._offset;
        this.skip(2);
        const value = this._dataView.getUint16(this._position + offset, true);
        // decode float16 value
        const s = (value & 0x8000) >> 15;
        const e = (value & 0x7C00) >> 10;
        const f = value & 0x03FF;
        if(e == 0) {
            return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
        }
        else if (e == 0x1F) {
            return f ? NaN : ((s ? -1 : 1) * Infinity);
        }
        return (s ? -1 : 1) * Math.pow(2, e-15) * (1 + (f / Math.pow(2, 10)));
    }

    float32() {
        const offset = this._offset;
        this.skip(4);
        return this._dataView.getFloat32(this._position + offset, true);
    }

    float64() {
        const offset = this._offset;
        this.skip(8);
        return this._dataView.getFloat64(this._position + offset, true);
    }

    string(size, encoding) {
        if (!size || size == -1) {
            let position = this._position + this._offset;
            while (this._buffer[position] != 0) {
                position++;
            }
            size = position - this._position + this._offset + 1;
        }
        const data = this.bytes(size);
        return hdf5.Reader.decode(data, encoding);
    }

    static decode(data, encoding) {
        let text = '';
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
            case 8: {
                const lo = this.uint32();
                const hi = this.uint32();
                if (lo === 0xffffffff && hi === 0xffffffff) {
                    return undefined;
                }
                return new long.Long(lo, hi, true).toNumber();
            }
            case 4: {
                const value = this.uint32();
                if (value === 0xffffffff) {
                    return undefined;
                }
                return value;
            }
        }
        throw new hdf5.Error('Unsupported offset size \'' + this._offsetSize + '\'.');
    }

    length() {
        switch (this._lengthSize) {
            case 8: {
                const lo = this.uint32();
                const hi = this.uint32();
                if (lo === 0xffffffff && hi === 0xffffffff) {
                    return undefined;
                }
                return new long.Long(lo, hi, true).toNumber();
            }
            case 4: {
                const value = this.uint32();
                if (value === 0xffffffff) {
                    return undefined;
                }
                return value;
            }
        }
        throw new hdf5.Error('Unsupported length size \'' + this._lengthSize + '\'.');
    }

    at(position) {
        let reader = new hdf5.Reader(null);
        reader._buffer = this._buffer;
        reader._dataView = this._dataView;
        reader._position = position;
        reader._offset = 0;
        reader._offsetSize = this._offsetSize;
        reader._lengthSize = this._lengthSize;
        return reader;
    }

    clone() {
        let reader =  new hdf5.Reader(this._buffer, this._position);
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
        const offset = this._offset;
        const buffer = this.bytes(text.length);
        for (let i = 0; i < text.length; i++) {
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
        const version = reader.byte();
        if (version == 1) {
            reader.skip(1);
            const entriesUsed = reader.uint16();
            this.entries = [];
            for (let i = 0; i < entriesUsed; i++) {
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
        const cacheType = reader.uint32();
        reader.skip(4); // Reserved
        switch (cacheType) {
            case 0:
                break;
            case 1: {
                const scratchReader = reader.clone();
                this.treeAddress = scratchReader.offset();
                this.heapAddress = scratchReader.offset();
                break;
            }
            default:
                throw new hdf5.Error('Unsupported cache type \'' + cacheType + '\'.');
        }
        reader.skip(16); // Scratch-pad space
    }
};

hdf5.DataObjectHeader = class {

    constructor(reader) {
        // https://support.hdfgroup.org/HDF5/doc/H5.format.html#ObjectHeader
        this.attributes = [];
        this.links = [];
        this.continuations = [];
        const version = reader.match('OHDR') ? reader.byte() : reader.byte();
        switch (version) {
            case 1: {
                reader.skip(1);
                const messageCount = reader.uint16();
                reader.uint32();
                const objectHeaderSize = reader.uint32();
                reader.align(8);
                let end = reader.position + objectHeaderSize;
                for (let i = 0; i < messageCount; i++) {
                    const messageType = reader.uint16();
                    const messageSize = reader.uint16();
                    const messageFlags = reader.byte();
                    reader.skip(3);
                    reader.align(8);
                    const next = this._readMessage(reader, messageType, messageSize, messageFlags);
                    if ((!next || reader.position >= end) && this.continuations.length > 0) {
                        const continuation = this.continuations.shift();
                        reader = reader.at(continuation.offset);
                        end = continuation.offset + continuation.length;
                    }
                    else {
                        reader.align(8);
                    }
                }
                break;
            }
            case 2: {
                const flags = reader.byte();
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
                const size = reader.uint(flags & 0x03);
                let next = true;
                let end = reader.position + size;
                while (next && reader.position < end) {
                    const messageType = reader.byte();
                    const messageSize = reader.uint16();
                    const messageFlags = reader.byte();
                    if (reader.position < end) {
                        if ((flags & 0x04) != 0) {
                            reader.uint16();
                        }
                        next = this._readMessage(reader, messageType, messageSize, messageFlags);
                    } 
                    if ((!next || reader.position >= end) && this.continuations.length > 0) {
                        const continuation = this.continuations.shift();
                        reader = reader.at(continuation.offset);
                        end = continuation.offset + continuation.length;
                        if (!reader.match('OCHK')) {
                            throw new hdf5.Error("Invalid continuation block signature.");
                        }
                        next = true;
                    }
                }
                break;
            }
            default: {
                throw new hdf5.Error('Unsupported data object header version \'' + version + '\'.');
            }
        }
    }

    _readMessage(reader, type, size, flags) {
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
            case 0x0004:
            case 0x0005: // Fill Value
                this.fillValue = new hdf5.FillValue(reader.clone(), type);
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
            case 0x000B: // Filter Pipeline
                this.filterPipeline = new hdf5.FilterPipeline(reader.clone());
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
            case 0x000E: // Object Modification Time (Old)
            case 0x0012: // Object Modification Time
                this.objectModificationTime = new hdf5.ObjectModificationTime(reader.clone(), type);
                break;
            case 0x0015: // Attribute Info
                this.attributeInfo = new hdf5.AttributeInfo(reader.clone());
                break;
            default:
                throw new hdf5.Error('Unsupported message type \'' + type + '\'.');
        }
        reader.skip(size);
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

hdf5.Dataspace = class {

    constructor(reader) {
        // https://support.hdfgroup.org/HDF5/doc/H5.format.html#DataspaceMessage
        this._sizes = [];
        const version = reader.byte();
        switch (version) {
            case 1:
                this._dimensions = reader.byte();
                this._flags = reader.byte();
                reader.skip(1);
                reader.skip(4);
                for (let i = 0; i < this._dimensions; i++) {
                    this._sizes.push(reader.length());
                }
                if ((this._flags & 0x01) != 0) {
                    this._maxSizes = [];
                    for (let j = 0; j < this._dimensions; j++) {
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
                for (let k = 0; k < this._dimensions; k++) {
                    this._sizes.push(reader.length());
                }
                if ((this._flags & 0x01) != 0) {
                    this._maxSizes = [];
                    for (let l = 0; l < this._dimensions; l++) {
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
        return this._readArray(datatype, reader, this._sizes, 0);
    }

    _readArray(datatype, reader, shape, dimension) {
        const array = [];
        const size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                array.push(datatype.read(reader));
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                array.push(this._readArray(datatype, reader, shape, dimension + 1));
            }
        }     
        return array;
    }

    decode(datatype, data, globalHeap) {
        if (this._dimensions == 0) {
            return datatype.decode(data, globalHeap);
        }
        return this._decodeArray(datatype, data, globalHeap, this._sizes, 0);
    }

    _decodeArray(datatype, data, globalHeap, shape, dimension) {
        const size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                data[i] = datatype.decode(data[i], globalHeap);
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                data[j] = this._decodeArray(datatype, data[j], shape, dimension + 1);
            }
        }
        return data;
    }
};

hdf5.LinkInfo = class {

    constructor(reader) {
        const version = reader.byte();
        switch (version) {
            case 0: {
                const flags = reader.byte();
                if ((flags & 1) != 0) {
                    this.maxCreationIndex = reader.uint64();
                }
                this.fractalHeapAddress = reader.offset();
                this.nameIndexTreeAddress = reader.offset();
                if ((flags & 2) != 0) {
                    this.creationOrderIndexTreeAddress = reader.offset();
                }
                break;
            }
            default:
                throw new hdf5.Error("Unsupported link info message version '" + version + "'.");
        }
    }
};

hdf5.Datatype = class {

    constructor(reader) {
        // https://support.hdfgroup.org/HDF5/doc/H5.format.html#DatatypeMessage
        const format = reader.byte();
        const version = format >> 4;
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
                if ((this._flags & 0xfff6) === 0) {
                    if ((this._flags && 0x08) !== 0) {
                        switch (this._size) {
                            case 1: return 'int8';
                            case 2: return 'int16';
                            case 4: return 'int32';
                            case 8: return 'int64';
                        }
                    }
                    else {
                        switch (this._size) {
                            case 1: return 'uint8';
                            case 2: return 'uint16';
                            case 4: return 'uint32';
                            case 8: return 'uint64';
                        }
                    }
                }
                break;
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

    get littleEndian() {
        switch (this._class) {
            case 0: // fixed-point
            case 1: // floating-point
                return (this.flags & 0x01) == 0;
        }
        return true;
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
            case 9: { // variable-length
                const globalHeapObject = globalHeap.get(data.globalHeapID);
                if (globalHeapObject != null) {
                    const characterSet = (this._flags >> 8) & 0x0f; 
                    switch (characterSet) {
                        case 0:
                            return hdf5.Reader.decode(globalHeapObject.data, 'ascii');
                        case 1:
                            return hdf5.Reader.decode(globalHeapObject.data, 'utf-8');
                    }
                    throw new hdf5.Error('Unsupported character encoding.');
                }
                break;
            }
            default:
                throw new hdf5.Error('Unsupported datatype class \'' + this._class + '\'.');
        }
        return null;
    }
};

hdf5.FillValue = class {

    constructor(reader, type) {
        // https://support.hdfgroup.org/HDF5/doc/H5.format.html#FillValueMessage
        switch (type) {
            case 0x0004: {
                const size = reader.uint32();
                this.data = reader.bytes(size);
                break;
            }
            case 0x0005:
            default: {
                const version = reader.byte();
                switch (version) {
                    case 1:
                    case 2: {
                        reader.byte();
                        reader.byte();
                        const valueDefined = reader.byte();
                        if (version === 1 || valueDefined === 1) {
                            const size = reader.uint32();
                            this.data = reader.bytes(size);
                        }
                        break;
                    }
                    default:
                        throw new hdf5.Error('Unsupported fill value version \'' + version + '\'.');
                }
                break;
            }
        }
    }
};

hdf5.Link = class {

    constructor(reader) {
        // https://support.hdfgroup.org/HDF5/doc/H5.format.html#FillValueMessage
        const version = reader.byte();
        switch (version) {
            case 1: {
                const flags = reader.byte();
                this.type = (flags & 0x08) != 0 ? reader.byte() : 0;
                if ((flags & 0x04) != 0) {
                    this.creationOrder = reader.uint32();
                }
                const encoding = ((flags & 0x10) != 0 && reader.byte() == 1) ? 'utf-8' : 'ascii';
                this.name = reader.string(reader.uint(flags & 0x03), encoding);
                switch (this.type) {
                    case 0: // hard link
                        this.objectHeaderAddress = reader.offset();
                        break;
                    case 1: // soft link
                        break;
                }
                break;
            }
            default:
                throw new hdf5.Error('Unsupported link message version \'' + version + '\'.');
        }
    }
};

hdf5.DataLayout = class {

    constructor(reader) {
        // https://support.hdfgroup.org/HDF5/doc/H5.format.html#LayoutMessage
        const version = reader.byte();
        switch (version) {
            case 1: 
            case 2: {
                this.dimensionality = reader.byte();
                this.layoutClass = reader.byte();
                reader.skip(5);
                switch (this.layoutClass) {
                    case 1:
                        this.address = reader.offset();
                        this.dimensionSizes = [];
                        for (let i = 0; i < this.dimensionality - 1; i++) {
                            this.dimensionSizes.push(reader.int32());
                        }
                        break;
                    case 2: // Chunked
                        this.address = reader.offset();
                        this.dimensionSizes = [];
                        for (let i = 0; i < this.dimensionality - 1; i++) {
                            this.dimensionSizes.push(reader.int32());
                        }
                        this.datasetElementSize = reader.int32();
                        break;
                    default:
                        throw new hdf5.Error('Unsupported data layout class \'' + this.layoutClass + '\'.');
                }
                break;
            }
            case 3: {
                this.layoutClass = reader.byte();
                switch (this.layoutClass) {
                    case 1: // Contiguous
                        this.address = reader.offset();
                        this.size = reader.length();
                        break;
                    case 2: // Chunked
                        this.dimensionality = reader.byte();
                        this.address = reader.offset();
                        this.dimensionSizes = [];
                        for (let i = 0; i < this.dimensionality - 1; i++) {
                            this.dimensionSizes.push(reader.int32());
                        }
                        this.datasetElementSize = reader.int32();
                        break;
                    case 0: // Compact
                    default:
                        throw new hdf5.Error('Unsupported data layout class \'' + this.layoutClass + '\'.');
                }
                break;
            }
            default:
                throw new hdf5.Error('Unsupported data layout version \'' + version + '\'.');
        }
    }
};

hdf5.GroupInfo = class {

    constructor(reader) {
        const version = reader.byte();
        switch (version) {
            case 0: {
                const flags = reader.byte();
                if ((flags & 0x01) != 0) {
                    this.maxCompactLinks = reader.uint16();
                    this.minDenseLinks = reader.uint16();
                }
                if ((flags & 0x02) != 0) {
                    this.estimatedEntriesNumber = reader.uint16();
                    this.estimatedLinkNameLengthEntires = reader.uint16();
                }
                break;
            }
            default:
                throw new hdf5.Error('Unsupported group info version \'' + version + '\'.');
        }
    }
};

hdf5.FilterPipeline = class {

    constructor(reader) {
        // https://support.hdfgroup.org/HDF5/doc/H5.format.html#FilterMessage
        const version = reader.byte();
        switch (version) {
            case 1: {
                this.filters = [];
                const numberOfFilters = reader.byte();
                reader.skip(2);
                reader.skip(4);
                for (let i = 0; i < numberOfFilters; i++) {
                    this.filters.push(new hdf5.Filter(reader));
                    reader.align(8);
                }
                break;
            }
            default:
                throw new hdf5.Error('Unsupported filter pipeline message version \'' + version + '\'.'); 
        }
    }
};

hdf5.Filter = class {

    constructor(reader) {
        this.id = reader.int16();
        const nameLength = reader.int16();
        this.flags = reader.int16();
        const clientDataSize = reader.int16();
        this.name = reader.string(nameLength, 'ascii');
        this.clientData = reader.bytes(clientDataSize * 4);
    }

    decode(data) {
        switch (this.id) {
            case 1: { // gzip
                const rawData = data.subarray(2, data.length); // skip zlib header
                return new zip.Inflater().inflateRaw(rawData);
            }
            default:
                throw hdf5.Error("Unsupported filter '" + this.name + "'.");
        }
    }
}

hdf5.Attribute = class {

    constructor(reader) {
        const version = reader.byte();
        switch (version) {
            case 1: {
                reader.skip(1);
                const nameSize = reader.uint16();
                const datatypeSize = reader.uint16();
                const dataspaceSize = reader.uint16();
                this.name = reader.string(nameSize, 'utf-8');
                reader.align(8);
                this._datatype = new hdf5.Datatype(reader.clone());
                reader.skip(datatypeSize);
                reader.align(8);
                this._dataspace = new hdf5.Dataspace(reader.clone());
                reader.skip(dataspaceSize);
                reader.align(8);
                this._data = this._dataspace.read(this._datatype, reader);
                break;
            }
            case 3: {
                reader.byte();
                const nameSize = reader.uint16();
                const datatypeSize = reader.uint16();
                const dataspaceSize = reader.uint16();
                const encoding = reader.byte() == 1 ? 'utf-8' : 'ascii';
                this.name = reader.string(nameSize, encoding);
                this._datatype = new hdf5.Datatype(reader.clone());
                reader.skip(datatypeSize);
                this._dataspace = new hdf5.Dataspace(reader.clone());
                reader.skip(dataspaceSize);
                this._data = this._dataspace.read(this._datatype, reader);
                break;
            }
            default:
                throw new hdf5.Error('Unsupported attribute message version \'' + version + '\'.'); 
        }
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

hdf5.ObjectModificationTime = class {

    constructor(reader, type) {
        // https://support.hdfgroup.org/HDF5/doc/H5.format.html#ModificationTimeMessage
        switch (type) {
            case 0x000E: {
                this.year = reader.uint32();
                this.month = reader.uint16();
                this.day = reader.uint16();
                this.hour = reader.uint16();
                this.minute = reader.uint16();
                this.second = reader.uint16();
                reader.skip(2);
                break;
            }
            case 0x0012: {
                const version = reader.byte();
                reader.skip(3);
                switch (version) {
                    case 1:
                        this.timestamp = reader.uint32();
                        break;
                    default:
                        throw new hdf5.Error('Unsupported object modification time message version \'' + version + '\'.'); 
                }
                break;
            }
        }
    }
}

hdf5.AttributeInfo = class {

    constructor(reader) {
        const version = reader.byte();
        switch (version) {
            case 0: {
                const flags = reader.byte();
                if ((flags & 1) != 0) {
                    this.maxCreationIndex = reader.uint64();
                }
                this.fractalHeapAddress = reader.offset();
                this.attributeNameTreeAddress = reader.offset();
                if ((flags & 2) != 0) {
                    this.attributeCreationOrderTreeAddress = reader.offset();
                }
                break;
            }
            default:
                throw new hdf5.Error('Unsupported attribute info message version \'' + version + '\'.'); 
        }
    }
};

hdf5.Tree = class {

    constructor(reader, dimensionality) {
        // https://support.hdfgroup.org/HDF5/doc/H5.format.html#V1Btrees
        if (!reader.match('TREE')) {
            throw new hdf5.Error("Not a valid 'TREE' block.");
        }
        this.type = reader.byte();
        this.level = reader.byte();
        const entriesUsed = reader.uint16();
        reader.offset(); // address of left sibling
        reader.offset(); // address of right sibling
        this.nodes = [];
        switch (this.type) {
            case 0: // Group nodes
                for (let i = 0; i < entriesUsed; i++) {
                    reader.length();
                    const childPointer = reader.offset();
                    if (this.level == 0) {
                        this.nodes.push(new hdf5.SymbolTableNode(reader.at(childPointer)));
                    }
                    else {
                        const tree = new hdf5.Tree(reader.at(childPointer));
                        this.nodes = this.nodes.concat(tree.nodes);
                    }
                }
                break;
            case 1: // Raw data chunk nodes
                for (let i = 0; i < entriesUsed; i++) {
                    const size = reader.int32();
                    const filterMask = reader.int32();
                    const fields = [];
                    for (let j = 0; j < dimensionality; j++) {
                        fields.push(reader.uint64())
                    }
                    const childPointer = reader.offset();
                    if (this.level == 0) {
                        const data = reader.at(childPointer).bytes(size);
                        this.nodes.push({ data: data, fields: fields, filterMask: filterMask });
                    }
                    else {
                        const tree = new hdf5.Tree(reader.at(childPointer), dimensionality);
                        this.nodes = this.nodes.concat(tree.nodes);
                    }
                }
                break;  
            default:
                throw new hdf5.Error('Unsupported B-Tree node type \'' + this.type + '\'.');
        }
    }
};

hdf5.Heap = class {

    constructor(reader) {
        this._reader = reader;
        if (!reader.match('HEAP')) {
            throw new hdf5.Error("Not a valid 'HEAP' block.");
        }
        const version = reader.byte();
        switch (version) {
            case 0: {
                reader.skip(3);
                this._dataSize = reader.length();
                this._offsetToHeadOfFreeList = reader.length();
                this._dataAddress = reader.offset();
                break;
            }
            default: {
                throw new hdf5.Error('Unsupported Local Heap version \'' + version + '\'.');
            }
        }
    }

    getString(offset) {
        const reader = this._reader.at(this._dataAddress + offset);
        return reader.string(-1, 'utf-8');
    }
};

hdf5.GlobalHeap = class {

    constructor(reader) {
        this._reader = reader; 
        this._collections = new Map();
    }

    get(globalHeapID) {
        const address = globalHeapID.address;
        if (!this._collections.has(address)) {
            this._collections.set(address, new hdf5.GlobalHeapCollection(this._reader.at(address)));
        }
        return this._collections.get(globalHeapID.address).getObject(globalHeapID.objectIndex);
    }
};

hdf5.GlobalHeapCollection = class {

    constructor(reader) {
        const startPosition = reader.position;
        if (!reader.match('GCOL')) {
            throw new hdf5.Error("Not a valid 'GCOL' block.");
        }
        const version = reader.byte();
        switch (version) {
            case 1: {
                reader.skip(3);
                this._objects = new Map();
                const size = reader.length();
                const endPosition = startPosition + size;
                while (reader.position < endPosition) {
                    const index = reader.uint16();
                    if (index == 0) {
                        break;
                    }
                    this._objects.set(index, new hdf5.GlobalHeapObject(reader));
                    reader.align(8);
                }
                break;
            }
            default: {
                throw new hdf5.Error('Unsupported global heap collection version \'' + version + '\'.');
            }
        }
    }

    getObject(objectIndex) {
        if (this._objects.has(objectIndex)) {
            return this._objects.get(objectIndex);
        }
        return null;
    }
};

hdf5.GlobalHeapObject = class {

    constructor(reader) {
        reader.uint16();
        reader.skip(4);
        this.data = reader.bytes(reader.length());
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