
import * as assert from 'node:assert/strict';
import * as base from '../source/base.js';

// A complex number stores the real component first in either byte order.
for (const littleEndian of [false, true]) {
    for (const [dataType, size, setter] of [
        ['complex<int32>', 4, 'setInt32'],
        ['complex<float16>', 2, 'setFloat16'],
        ['complex<float32>', 4, 'setFloat32'],
        ['complex<float64>', 8, 'setFloat64']
    ]) {
        const data = new Uint8Array(4 * size);
        const view = new DataView(data.buffer);
        const values = [[1, -2], [-3, 4]];
        for (let i = 0; i < values.length; i++) {
            view[setter](i * 2 * size, values[i][0], littleEndian);
            view[setter]((i * 2 + 1) * size, values[i][1], littleEndian);
        }
        if (dataType !== 'complex<int32>') {
            const encoded = new Uint8Array(data.length);
            const output = new DataView(encoded.buffer);
            const complexSetter = setter.replace('set', 'setComplex');
            for (let i = 0; i < values.length; i++) {
                output[complexSetter](i * 2 * size, new base.Complex(...values[i]), littleEndian);
            }
            assert.deepStrictEqual(encoded, data, `${complexSetter} ${littleEndian}`);
        }
        const tensor = new base.Tensor({
            encoding: littleEndian ? '<' : '>',
            type: { dataType, shape: { dimensions: [2] } },
            values: data
        });
        assert.deepStrictEqual(tensor.value.map((value) => [value.real, value.imaginary]), values, `${dataType} ${tensor.encoding}`);
    }
}
