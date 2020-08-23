/* jshint esversion: 6 */

var numpy = numpy || {};

numpy.Array = class {

    constructor(buffer) {
        if (buffer) {
            const reader = new numpy.Reader(buffer);
            const signature = [ 0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59 ];
            if (!reader.bytes(6).every((v, i) => v == signature[i])) {
                throw new numpy.Error('Invalid signature.');
            }
            const major = reader.byte();
            const minor = reader.byte();
            if (major !== 1 && minor !== 0) {
                throw new numpy.Error("Invalid version '" + [ major, minor ].join('.') + "'.");
            }
            const header = JSON.parse(reader.string().trim().replace(/'/g, '"').replace("False", "false").replace("(", "[").replace(/,*\),*/g, "]"));
            if (header.fortran_order) {
                throw new numpy.Error("Fortran order is not supported.'");
            }
            if (!header.descr || header.descr.length < 2) {
                throw new numpy.Error("Missing property 'descr'.");
            }
            if (!header.shape) {
                throw new numpy.Error("Missing property 'shape'.");
            }
            this._shape = header.shape;
            this._byteOrder = header.descr[0];
            switch (this._byteOrder) {
                case '|': {
                    this._dataType = header.descr.substring(1);
                    this._data = reader.bytes(reader.size - reader.position);
                    break;
                }
                case '>':
                case '<': {
                    if (header.descr.length !== 3) {
                        throw new numpy.Error("Unsupported data type '" + header.descr + "'.");
                    }
                    this._dataType = header.descr.substring(1);
                    const size = parseInt(header.descr[2], 10) * this._shape.reduce((a, b) => a * b, 1);
                    this._data = reader.bytes(size);
                    break;
                }
                default:
                    throw new numpy.Error("Unsupported data type '" + header.descr + "'.");
            }
        }
    }

    get data() {
        return this._data;
    }

    set data(value) {
        this._data = value;
    }

    get dataType() {
        return this._dataType;
    }

    set dataType(value) {
        this._dataType = value;
    }

    get shape() {
        return this._shape;
    }

    set shape(value) {
        this._shape = value;
    }

    get byteOrder() {
        return this._byteOrder;
    }

    set byteOrder(value) {
        this._byteOrder = value;
    }

    toBuffer() {

        const writer = new numpy.Writer();

        writer.bytes([ 0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59 ]); // '\\x93NUMPY'
        writer.byte(1); // major
        writer.byte(0); // minor

        const context = {
            itemSize: 1,
            position: 0,
            dataType: this._dataType,
            byteOrder: this._byteOrder || '<',
            shape: this._shape,
            descr: '',
        };

        if (context.byteOrder !== '<' && context.byteOrder !== '>') {
            throw new numpy.Error("Unknown byte order '" + this._byteOrder + "'.");
        }
        if (context.dataType.length !== 2 || (context.dataType[0] !== 'f' && context.dataType[0] !== 'i' && context.dataType[0] !== 'u')) {
            throw new numpy.Error("Unsupported data type '" + this._dataType + "'.");
        }

        context.itemSize = parseInt(context.dataType[1], 10);

        let shape = '';
        switch (this._shape.length) {
            case 0:
                throw new numpy.Error('Invalid shape.');
            case 1:
                shape = '(' + this._shape[0].toString() + ',)';
                break;
            default:
                shape = '(' + this._shape.map((dimension) => dimension.toString()).join(', ') + ')';
                break;
        }

        const properties = [
            "'descr': '" + context.byteOrder + context.dataType + "'",
            "'fortran_order': False",
            "'shape': " + shape
        ];
        let header = '{ ' + properties.join(', ') + ' }';
        header += ' '.repeat(16 - ((header.length + 2 + 8 + 1) & 0x0f)) + '\n';
        writer.string(header);

        const size = context.itemSize * this._shape.reduce((a, b) => a * b);
        context.data = new Uint8Array(size);
        context.view = new DataView(context.data.buffer, context.data.byteOffset, size);
        numpy.Array._encodeDimension(context, this._data, 0);
        writer.bytes(context.data);

        return writer.toBuffer();
    }

    static _encodeDimension(context, data, dimension) {
        const size = context.shape[dimension];
        const littleEndian = context.byteOrder === '<';
        if (dimension == context.shape.length - 1) {
            for (let i = 0; i < size; i++) {
                switch (context.dataType) {
                    case 'f2':
                        context.view.setFloat16(context.position, data[i], littleEndian);
                        break;
                    case 'f4':
                        context.view.setFloat32(context.position, data[i], littleEndian);
                        break;
                    case 'f8':
                        context.view.setFloat64(context.position, data[i], littleEndian);
                        break;
                    case 'i1':
                        context.view.setInt8(context.position, data[i], littleEndian);
                        break;
                    case 'i2':
                        context.view.setInt16(context.position, data[i], littleEndian);
                        break;
                    case 'i4':
                        context.view.setInt32(context.position, data[i], littleEndian);
                        break;
                    case 'i8':
                        context.view.setInt64(context.position, data[i], littleEndian);
                        break;
                    case 'u1':
                        context.view.setUint8(context.position, data[i], littleEndian);
                        break;
                    case 'u2':
                        context.view.setUint16(context.position, data[i], littleEndian);
                        break;
                    case 'u4':
                        context.view.setUint32(context.position, data[i], littleEndian);
                        break;
                    case 'u8':
                        context.view.setUint64(context.position, data[i], littleEndian);
                        break;
                }
                context.position += context.itemSize;
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                numpy.Array._encodeDimension(context, data[j], dimension + 1);
            }
        }
    }
};

numpy.Reader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._position = 0;
    }

    get position() {
        return this._position;
    }

    get size() {
        return this._buffer.length;
    }

    byte() {
        return this._buffer[this._position++];
    }

    bytes(size) {
        const value = this._buffer.slice(this._position, this._position + size);
        this._position += size;
        return value;
    }

    uint16() {
        return this.byte() | (this.byte() << 8);
    }

    string() {
        const size = this.uint16();
        let value = '';
        for (let i = 0; i < size; i++) {
            value += String.fromCharCode(this.byte());
        }
        return value;
    }
};

numpy.Writer = class {

    constructor() {
        this._length = 0;
        this._head = null;
        this._tail = null;
    }

    byte(value) {
        this.bytes([ value ]);
    }

    uint16(value) {
        this.bytes([ value & 0xff, (value >> 8) & 0xff ]);
    }

    bytes(values) {
        const array = new Uint8Array(values.length);
        for (let i = 0; i < values.length; i++) {
            array[i] = values[i];
        }
        this._write(array);
    }

    string(value) {
        this.uint16(value.length);
        const array = new Uint8Array(value.length);
        for (let i = 0; i < value.length; i++) {
            array[i] = value.charCodeAt(i);
        }
        this._write(array);
    }

    _write(array) {
        const node = { buffer: array, next: null };
        if (this._tail) {
            this._tail.next = node;
        }
        else {
            this._head = node;
        }
        this._tail = node;
        this._length += node.buffer.length;
    }

    toBuffer() {
        const array = new Uint8Array(this._length);
        let position = 0;
        let head = this._head;
        while (head != null) {
            array.set(head.buffer, position);
            position += head.buffer.length;
            head = head.next;
        }
        return array;
    }
};

numpy.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'NumPy Error';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Array = numpy.Array;
}