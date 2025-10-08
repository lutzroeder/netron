// Minimal smoke test for parser-template.js
(async () => {
    try {
        const template = await import('../source/parser-template.js');
        const ModelFactory = template.default && template.default.ModelFactory;
        if (!ModelFactory) {
            throw new Error('ModelFactory export missing');
        }
        const factory = new ModelFactory();

        // non-matching identifier
        const fakeContext1 = { identifier: 'model.onnx', value: null, set: () => null };
        const result1 = await factory.match(fakeContext1);
        if (result1 !== null) {
            throw new Error('Expected null for non-matching identifier');
        }

        // matching by extension
        const fakeContext2 = { identifier: 'something.tpl', value: null, set: (k, v) => v };
        const result2 = await factory.match(fakeContext2);
        if (!result2 || result2.name !== 'template') {
            throw new Error('Expected template reader for .tpl identifier');
        }

        // create a small binary buffer with magic 'TPL\0', major=1, minor=2, reserved=0x0000
        const buf = new Uint8Array([0x54, 0x50, 0x4C, 0x00, 0x01, 0x02, 0x00, 0x00]);
        const fakeContext3 = { identifier: 'buffer', value: buf, set: (k, v) => v };
        // match should detect by magic as well
        const result3 = await factory.match(fakeContext3);
        if (!result3) {
            throw new Error('Expected template reader for magic buffer');
        }
        // open and verify header parsing
        const model = await factory.open(fakeContext3);
        if (!model || !model.header || model.header.magic !== 'TPL\u0000' || model.header.version !== '1.2') {
            throw new Error('Header parsing failed: ' + JSON.stringify(model && model.header));
        }
        console.log('parser-template test: PASS');
    } catch (err) {
        console.error('parser-template test: FAIL', err && err.message ? err.message : err);
        process.exit(1);
    }
})();
