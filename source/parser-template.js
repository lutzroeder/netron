// Minimal parser template for Netron
// Follow the ModelFactory pattern: export a class with match(context), open(context), and optional filter(context, type).

import * as base from './base.js';

const template = {};

template.ModelFactory = class {

    async match(context) {
        // Return a reader object or null.
        const identifier = context && context.identifier ? context.identifier.toLowerCase() : '';
        if (identifier.endsWith('.tpl')) {
            return context.set('template.reader', { name: 'template', format: 'template' });
        }
        // Also attempt to detect by magic bytes at the start of a BinaryStream
        try {
            const value = context && context.value ? context.value : null;
            const buffer = value && (value instanceof Uint8Array || value.peek) ? (value instanceof Uint8Array ? value : value.peek(8)) : null;
            if (buffer && buffer.length >= 4) {
                // magic 'TPL\0'
                if (buffer[0] === 0x54 && buffer[1] === 0x50 && buffer[2] === 0x4C && buffer[3] === 0x00) {
                    return context.set('template.reader', { name: 'template', format: 'template' });
                }
            }
        } catch (e) {
            // ignore detection errors
        }
        return null;
    }

    async open(context) {
        // Expect context.value to be a Uint8Array or a stream-like object compatible with base.BinaryReader.open
        const data = context && context.value ? context.value : null;
        let reader = null;
        try {
            reader = base.BinaryReader.open(data, true);
        } catch (e) {
            // fallback: if data is not directly readable, try peek()
            if (data && typeof data.peek === 'function') {
                const buf = data.peek(16);
                reader = base.BinaryReader.open(buf, true);
            } else {
                throw e;
            }
        }

        // Read a tiny header: 4-byte magic, 1-byte major, 1-byte minor, 2-byte reserved
        const magicBytes = reader.read(4);
        const magic = String.fromCharCode.apply(null, magicBytes instanceof Uint8Array ? Array.from(magicBytes) : magicBytes);
        const major = reader.byte();
        const minor = reader.byte();
        // We don't need the reserved bytes, but advance stream
        reader.skip(2);

        return {
            format: 'template',
            producer: 'parser-template',
            header: { magic, version: `${major}.${minor}` },
            metadata: new base.Metadata()
        };
    }

    filter(context, type) {
        return true;
    }
};

export default template;
