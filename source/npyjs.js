
const dtypes = {
    "<u1": {
        name: "uint8",
        size: 8,
        arrayConstructor: Uint8Array,
    },
    "|u1": {
        name: "uint8",
        size: 8,
        arrayConstructor: Uint8Array,
    },
    "<u2": {
        name: "uint16",
        size: 16,
        arrayConstructor: Uint16Array,
    },
    "|i1": {
        name: "int8",
        size: 8,
        arrayConstructor: Int8Array,
    },
    "<i2": {
        name: "int16",
        size: 16,
        arrayConstructor: Int16Array,
    },
    "<u4": {
        name: "uint32",
        size: 32,
        arrayConstructor: Int32Array,
    },
    "<i4": {
        name: "int32",
        size: 32,
        arrayConstructor: Int32Array,
    },
    "<u8": {
        name: "uint64",
        size: 64,
        arrayConstructor: BigUint64Array,
    },
    "<i8": {
        name: "int64",
        size: 64,
        arrayConstructor: BigInt64Array,
    },
    "<f4": {
        name: "float32",
        size: 32,
        arrayConstructor: Float32Array
    },
    "<f8": {
        name: "float64",
        size: 64,
        arrayConstructor: Float64Array
    },
};

const npyjs = {
    parse: (arrayBuffer) => {
        const headerLength = new DataView(arrayBuffer.slice(8, 10)).getUint8(0);
        const offsetBytes = 10 + headerLength;

        const hcontents = new TextDecoder("utf-8").decode(
            new Uint8Array(arrayBuffer.slice(10, 10 + headerLength))
        );
        const header = JSON.parse(
            hcontents
                .toLowerCase() // True -> true
                .replace(/'/g, '"')
                .replace("(", "[")
                .replace(/,*\),*/g, "]")
        );
        const shape = header.shape;
        const dtype = dtypes[header.descr];
        const nums = new dtype["arrayConstructor"](
            arrayBuffer,
            offsetBytes
        );
        return {
            dtype: dtype.name,
            data: nums,
            shape,
            fortranOrder: header.fortran_order
        };
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports = npyjs;
}