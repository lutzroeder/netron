
// Experimental

var weka = weka || {};
var java = {};

weka.ModelFactory = class {

    match(context) {
        try {
            const stream = context.stream;
            if (stream.length >= 5) {
                const signature = [ 0xac, 0xed ];
                if (stream.peek(2).every((value, index) => value === signature[index])) {
                    const reader = new java.io.InputObjectStream(stream);
                    const obj = reader.read();
                    if (obj && obj.$class && obj.$class.name) {
                        return 'weka';
                    }
                }
            }
        } catch (err) {
            // continue regardless of error
        }
        return undefined;
    }

    async open(context) {
        const reader = new java.io.InputObjectStream(context.stream);
        const obj = reader.read();
        throw new weka.Error("Unsupported type '" + obj.$class.name + "'.");
    }
};

weka.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Weka model.';
    }
};

java.io = {};

java.io.InputObjectStream = class {

    constructor(stream) {
        // Object Serialization Stream Protocol
        // https://www.cis.upenn.edu/~bcpierce/courses/629/jdkdocs/guide/serialization/spec/protocol.doc.html
        if (stream.length < 5) {
            throw new java.io.Error('Invalid stream size');
        }
        const signature = [ 0xac, 0xed ];
        if (!stream.peek(2).every((value, index) => value === signature[index])) {
            throw new java.io.Error('Invalid stream signature');
        }
        this._reader = new java.io.InputObjectStream.BinaryReader(stream.peek());
        this._references = [];
        this._reader.skip(2);
        const version = this._reader.uint16();
        if (version !== 0x0005) {
            throw new java.io.Error("Unsupported version '" + version + "'.");
        }
    }

    read() {
        return this._object();
    }

    _object() {
        const code = this._reader.byte();
        switch (code) {
            case 0x73: { // TC_OBJECT
                const obj = {};
                obj.$class = this._classDesc();
                this._newHandle(obj);
                this._classData(obj);
                return obj;
            }
            case 0x74: { // TC_STRING
                return this._newString(false);
            }
            default: {
                throw new java.io.Error("Unsupported code '" + code + "'.");
            }
        }
    }

    _classDesc() {
        const code = this._reader.byte();
        switch (code) {
            case 0x72: // TC_CLASSDESC
                this._reader.skip(-1);
                return this._newClassDesc();
            case 0x71: // TC_REFERENCE
                return this._references[this._reader.uint32() - 0x7e0000];
            case 0x70: // TC_NULL
                this._reader.byte();
                return null;
            default:
                throw new java.io.Error("Unsupported code '" + code + "'.");
        }
    }

    _newClassDesc() {
        const code = this._reader.byte();
        switch (code) {
            case 0x72: { // TC_CLASSDESC
                const classDesc = {};
                classDesc.name = this._reader.string();
                classDesc.id = this._reader.uint64().toString();
                this._newHandle(classDesc);
                classDesc.flags = this._reader.byte();
                classDesc.fields = [];
                const count = this._reader.uint16();
                for (let i = 0; i < count; i++) {
                    const field = {};
                    field.type = String.fromCharCode(this._reader.byte());
                    field.name = this._reader.string();
                    if (field.type === '[' || field.type === 'L') {
                        field.classname = this._object();
                    }
                    classDesc.fields.push(field);
                }
                if (this._reader.byte() !== 0x78) {
                    throw new java.io.Error('Expected TC_ENDBLOCKDATA.');
                }
                classDesc.superClass = this._classDesc();
                return classDesc;
            }
            case 0x7D: // TC_PROXYCLASSDESC
                return null;
            default:
                throw new java.io.Error("Unsupported code '" + code + "'.");
        }
    }

    _classData(/* obj */) {
        /*
        const classname = obj.$class.name;
        let flags = obj.$class.flags;
        let superClass = obj.$class.superClass;
        while (superClass) {
            flags |= superClass.flags;
            superClass = superClass.superClass;
        }
        if (flags & 0x02) { // SC_SERIALIZABLE
            debugger;
            var customObject = objects[classname];
            var hasReadObjectMethod = customObject && customObject.readObject;
            if (flags & 0x01) { // SC_WRITE_METHOD
                if (!hasReadObjectMethod) {
                    throw new Error('Class "'+ classname + '" dose not implement readObject()');
                }
                customObject.readObject(this, obj);
                if (this._reader.byte() !== 0x78) { // TC_ENDBLOCKDATA
                    throw new java.io.Error('Expected TC_ENDBLOCKDATA.');
                }
            }
            else {
                if (hasReadObjectMethod) {
                    customObject.readObject(this, obj);
                    if (this._reader.byte() !== 0x78) { // TC_ENDBLOCKDATA
                        throw new java.io.Error('Expected TC_ENDBLOCKDATA.');
                    }
                }
                else {
                    this._nowrclass(obj);
                }
            }
        }
        else if (flags & 0x04) { // SC_EXTERNALIZABLE
            if (flags & 0x08) { // SC_BLOCK_DATA
                this._objectAnnotation(obj);
            }
            else {
                this._externalContents();
            }
        }
        else {
            throw new Error('Illegal flags: ' + flags);
        }
        */
    }

    _newString(long) {
        const value = this._reader.string(long);
        this._newHandle(value);
        return value;
    }

    _newHandle(obj) {
        this._references.push(obj);
    }
};

java.io.InputObjectStream.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._position = 0;
        this._length = buffer.length;
        this._view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._end) {
            throw new java.io.Error('Expected ' + (this._position - this._end) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._buffer[position];
    }

    uint16() {
        const position = this._position;
        this.skip(2);
        return this._view.getUint16(position, false);
    }

    uint32() {
        const position = this._position;
        this.skip(4);
        return this._view.getUint32(position, false);
    }

    uint64() {
        const position = this._position;
        this.skip(8);
        return this._view.getUint64(position, false);
    }

    string(long) {
        const size = long ? this.uint64().toNumber() : this.uint16();
        const position = this._position;
        this.skip(size);
        this._decoder = this._decoder || new TextDecoder('utf-8');
        return this._decoder.decode(this._buffer.subarray(position, this._position));
    }
};

java.io.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Object Serialization Stream Protocol.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = weka.ModelFactory;
}