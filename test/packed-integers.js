import * as base from '../source/base.js';
import assert from 'node:assert/strict';

// Pack with BigInt as an independent oracle, including values crossing three
// or more bytes and a final element without any extra padding bytes.
for (let bits = 1; bits <= 32; bits++) {
    const sign = 2 ** (bits - 1);
    const unsigned = [0, 1, sign - 1, sign, 2 ** bits - 1, 3 % (2 ** bits), (sign + 1) % (2 ** bits)];
    const signed = unsigned.map((value) => value >= sign ? value - 2 ** bits : value);
    for (const littleEndian of [false, true]) {
        let packed = 0n;
        for (let i = 0; i < unsigned.length; i++) {
            const shift = littleEndian ? i * bits : (unsigned.length - i - 1) * bits;
            packed |= BigInt(unsigned[i]) << BigInt(shift);
        }
        const length = Math.ceil(unsigned.length * bits / 8);
        if (!littleEndian) {
            packed <<= BigInt(length * 8 - unsigned.length * bits);
        }
        const bytes = Uint8Array.from({ length }, (_, i) => {
            const shift = littleEndian ? i * 8 : (length - i - 1) * 8;
            return Number((packed >> BigInt(shift)) & 255n);
        });
        const view = new DataView(bytes.buffer);
        for (let i = 0; i < unsigned.length; i++) {
            assert.equal(view.getUintBits(i, bits, littleEndian), unsigned[i], `uint${bits}[${i}], LE=${littleEndian}`);
            assert.equal(view.getIntBits(i, bits, littleEndian), signed[i], `int${bits}[${i}], LE=${littleEndian}`);
        }
        for (const [prefix, expected] of [['uint', unsigned], ['int', signed]]) {
            const tensor = new base.Tensor({
                type: { dataType: `${prefix}${bits}`, shape: { dimensions: [unsigned.length] } },
                encoding: littleEndian ? '<' : '>',
                values: bytes
            });
            assert.deepEqual(tensor.value, expected);
        }
    }
}
