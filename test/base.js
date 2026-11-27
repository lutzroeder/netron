
import * as assert from 'node:assert/strict';

/* eslint-disable no-extend-native */

// Exercise the fallback even when Node implements Float16.
const descriptor = Object.getOwnPropertyDescriptor(DataView.prototype, 'setFloat16');
Object.defineProperty(DataView.prototype, 'setFloat16', { value: undefined, configurable: true, writable: true });
await import('../source/base.js');
try {
    const cases = [
        [0, 0x0000], [-0, 0x8000], [1, 0x3c00],
        [1.0009, 0x3c01], [-1.0009, 0xbc01],
        [1 + 2 ** -11, 0x3c00],
        [1 + 3 * 2 ** -11, 0x3c02],
        [1 + 2 ** -11 + 2 ** -30, 0x3c01],
        [1 + 2 ** -11 - 2 ** -30, 0x3c00],
        [2 ** -25, 0x0000],
        [2 ** -25 + 2 ** -54, 0x0001],
        [2 ** -24, 0x0001],
        [65504, 0x7bff], [65519, 0x7bff], [65520, 0x7c00],
        [Infinity, 0x7c00], [-Infinity, 0xfc00]
    ];
    for (const littleEndian of [false, true]) {
        const view = new DataView(new ArrayBuffer(4));
        for (const [value, bits] of cases) {
            view.setFloat16(1, value, littleEndian);
            assert.strictEqual(view.getUint16(1, littleEndian), bits, `${value} ${littleEndian}`);
        }
        view.setFloat16(1, NaN, littleEndian);
        assert.ok(Number.isNaN(view.getFloat16(1, littleEndian)));
    }
} finally {
    if (descriptor) {
        Object.defineProperty(DataView.prototype, 'setFloat16', descriptor);
    }
}

/* eslint-enable no-extend-native */
