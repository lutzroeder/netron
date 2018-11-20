/*jshint esversion: 6 */

var numpy = numpy || {};

numpy.Array = class {

    constructor(data, dataType, shape) {
        this._data = data;
        this._dataType = dataType;
        this._shape = shape;
    }

    toBuffer() {

        var writer = new numpy.Writer();

        writer.write([ 0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59 ]); // '\\x93NUMPY'
        writer.writeByte(1); // major
        writer.writeByte(0); // minor

        var context = {};
        context.itemSize = 1;
        context.position = 0;
        context.shape = this._shape;
        context.descr = '';

        switch (this._dataType) {
            case 'float16':
                context.itemSize = 2;
                context.descr = '<f2';
                break;
            case 'float32':
                context.itemSize = 4;
                context.descr = '<f4';
                break;
            case 'float64':
                context.itemSize = 8;
                context.descr = '<f8';
                break;
            case 'int8':
                context.itemSize = 1;
                context.descr = '<i1';
                break;
            case 'int16':
                context.itemSize = 2;
                context.descr = '<i2';
                break;
            case 'int32':
                context.itemSize = 4;
                context.descr = '<i4';
                break;
            case 'int64':
                context.itemSize = 8;
                context.descr = '<i8';
                break;
            case 'byte':
            case 'uint8':
                context.itemSize = 1;
                context.descr = '<u1';
                break;
            case 'uint16':
                context.itemSize = 2;
                context.descr = '<u2';
                break;
            case 'uint32':
                context.itemSize = 4;
                context.descr = '<u4';
                break;
            case 'uint64':
                context.itemSize = 8;
                context.descr = '<u8';
                break;
            default:
                throw new numpy.Error("Unknown data type '" + this._dataType + "'.");
        }

        var shape = '';
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

        var properties = [];
        properties.push("'descr': '" + context.descr + "'");
        properties.push("'fortran_order': False");
        properties.push("'shape': " + shape);
        var header = '{ ' + properties.join(', ') + ' }';
        header += ' '.repeat(16 - ((header.length + 2 + 8 + 1) & 0x0f)) + '\n';
        writer.writeUint16(header.length); // header size
        writer.writeString(header);

        var size = context.itemSize;
        this._shape.forEach((dimension) => {
            size *= dimension;
        });

        context.data = new Uint8Array(size);
        context.dataView = new DataView(context.data.buffer, context.data.byteOffset, size);
        numpy.Array._encodeDimension(context, this._data, 0);
        writer.write(context.data);

        return writer.toBuffer();
    }

    static _encodeDimension(context, data, dimension) {
        var size = context.shape[dimension];
        if (dimension == context.shape.length - 1) {
            for (var i = 0; i < size; i++) {
                switch (context.descr)
                {
                    case '<f2':
                        context.dataView.setFloat16(context.position, data[i], true);
                        break;
                    case '<f4':
                        context.dataView.setFloat32(context.position, data[i], true);
                        break;
                    case '<f8':
                        context.dataView.setFloat64(context.position, data[i], true);
                        break;
                    case '<i1':
                        context.dataView.setInt8(context.position, data[i], true);
                        break;
                    case '<i2':
                        context.dataView.setInt16(context.position, data[i], true);
                        break;
                    case '<i4':
                        context.dataView.setInt32(context.position, data[i], true);
                        break;
                    case '<i8':
                        context.data.set(data[i].toBuffer(), context.position);
                        break;
                    case '<u1':
                        context.dataView.setUint8(context.position, data[i], true);
                        break;
                    case '<u2':
                        context.dataView.setUint16(context.position, data[i], true);
                        break;
                    case '<u4':
                        context.dataView.setUint32(context.position, data[i], true);
                        break;
                    case '<u8':
                        context.data.set(data[i].toBuffer(), context.position);
                        break;
                }
                context.position += context.itemSize;
            }
        }
        else {
            for (var j = 0; j < size; j++) {
                numpy.Array._encodeDimension(context, data[j], dimension + 1);
            }
        }
    }
};

numpy.Writer = class {

    constructor() {
        this._length = 0;
        this._head = null;
        this._tail = null;
    }

    writeByte(value) {
        this.writeBytes([ value ]);
    }

    writeUint16(value) {
        this.writeBytes([ value & 0xff, (value >> 8) & 0xff ]);
    }

    writeBytes(values) {
        var array = new Uint8Array(values.length);
        for (var i = 0; i < values.length; i++) {
            array[i] = values[i];
        }
        this.write(array);
    }

    writeString(value) {
        var array = new Uint8Array(value.length);
        for (var i = 0; i < value.length; i++) {
            array[i] = value.charCodeAt(i);
        }
        this.write(array);
    }

    write(array) {
        var node = { buffer: array, next: null };
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
        var array = new Uint8Array(this._length);
        var position = 0;
        var head = this._head;
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