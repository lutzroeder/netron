
import * as fs from 'fs/promises';
import * as path from 'path';
import * as url from 'url';

const dirname = path.dirname(url.fileURLToPath(import.meta.url));
const rootDir = path.resolve(dirname, '..');
const constantsPath = path.join(rootDir, 'third_party/source/llama.cpp/gguf-py/gguf/constants.py');
const metadataPath = path.join(rootDir, 'source/gguf-metadata.json');

const overlayKeys = ['type', 'category', 'tensors', 'attributes', 'position_encoding', 'has_bias', 'has_qk_norm'];

const parseConstants = (text) => {
    const block = (name) => {
        const re = new RegExp(`^${name}[^=]*=\\s*\\{([\\s\\S]*?)^\\}`, 'm');
        const m = text.match(re);
        return m ? m[1] : '';
    };
    const archNames = new Map();
    for (const m of block('MODEL_ARCH_NAMES').matchAll(/MODEL_ARCH\.(\w+)\s*:\s*"([^"]+)"/g)) {
        archNames.set(m[1], m[2]);
    }
    const tensorNames = new Map();
    for (const m of block('TENSOR_NAMES').matchAll(/MODEL_TENSOR\.(\w+)\s*:\s*"([^"]+)"/g)) {
        tensorNames.set(m[1], m[2]);
    }
    const modelTensors = new Map();
    const mtBlock = block('MODEL_TENSORS');
    for (const am of mtBlock.matchAll(/MODEL_ARCH\.(\w+)\s*:\s*\[([\s\S]*?)\],/g)) {
        const items = [];
        for (const im of am[2].matchAll(/MODEL_TENSOR\.(\w+)/g)) {
            items.push(im[1]);
        }
        modelTensors.set(am[1], items);
    }
    return { archNames, tensorNames, modelTensors };
};

const classify = (template) => {
    const strip = (prefix) => template.startsWith(prefix) ? template.slice(prefix.length) : null;
    let bare = strip('blk.{bid}.');
    if (bare !== null) {
        return { section: 'blocks', bare };
    }
    bare = strip('enc.blk.{bid}.');
    if (bare !== null) {
        return { section: 'encoder.blocks', bare };
    }
    bare = strip('dec.blk.{bid}.');
    if (bare !== null) {
        return { section: 'decoder.blocks', bare };
    }
    bare = strip('enc.');
    if (bare !== null) {
        return { section: bare.startsWith('output') ? 'encoder.output' : 'encoder.input', bare };
    }
    bare = strip('dec.');
    if (bare !== null) {
        return { section: bare.startsWith('output') ? 'decoder.output' : 'decoder.input', bare };
    }
    if (template.startsWith('output')) {
        return { section: 'output', bare: template };
    }
    return { section: 'input', bare: template };
};

const buildEntry = (groupName, members, overlay) => {
    const entry = { name: groupName };
    if (overlay) {
        const rest = Object.keys(overlay).filter((k) => k !== 'name' && !overlayKeys.includes(k));
        for (const key of [...overlayKeys, ...rest]) {
            if (overlay[key] !== undefined) {
                entry[key] = overlay[key];
            }
        }
    }
    const existing = entry.tensors || [];
    const upstream = new Set(members);
    // Preserve curator-only aliases that carry a placeholder pattern (e.g.
    // `ffn_gate.{N}` for MoE per-expert tensors that upstream doesn't enumerate).
    const tensors = [
        ...existing.filter((t) => upstream.has(t) || t.includes('{')),
        ...members.filter((t) => !existing.includes(t))
    ];
    if (tensors.length > 1 || (tensors.length === 1 && tensors[0] !== groupName)) {
        entry.tensors = tensors;
    } else {
        delete entry.tensors;
    }
    return entry;
};

const generate = (archName, tensorList, tensorNames, existing) => {
    const groupOf = new Map();
    const overlays = new Map();
    const sectionOrder = new Map();
    // Track where the curator placed each group so a deliberate section move
    // survives regeneration (e.g. LFM2 stores its output norm under the tensor
    // name `token_embd_norm`, but the curator places it in `output`).
    const curatorSection = new Map();
    const ingest = (sectionName, list) => {
        if (!Array.isArray(list)) {
            return;
        }
        const order = [];
        const sectionOverlays = new Map();
        for (const entry of list) {
            groupOf.set(entry.name, entry.name);
            for (const t of entry.tensors || []) {
                groupOf.set(t, entry.name);
            }
            sectionOverlays.set(entry.name, entry);
            curatorSection.set(entry.name, sectionName);
            order.push(entry.name);
        }
        overlays.set(sectionName, sectionOverlays);
        sectionOrder.set(sectionName, order);
    };
    if (existing && existing.graph) {
        for (const key of ['input', 'blocks', 'output']) {
            ingest(key, existing.graph[key]);
        }
        for (const sub of ['encoder', 'decoder']) {
            if (existing.graph[sub]) {
                for (const key of ['input', 'blocks', 'output']) {
                    ingest(`${sub}.${key}`, existing.graph[sub][key]);
                }
            }
        }
    }
    const sectionGroups = new Map();
    const upstreamOrder = new Map();
    for (const enumKey of tensorList) {
        const template = tensorNames.get(enumKey);
        if (!template) {
            continue;
        }
        const { section: defaultSection, bare } = classify(template);
        const group = groupOf.get(bare) || bare;
        const section = curatorSection.get(group) || defaultSection;
        if (!sectionGroups.has(section)) {
            sectionGroups.set(section, new Map());
            upstreamOrder.set(section, []);
        }
        const sg = sectionGroups.get(section);
        if (!sg.has(group)) {
            sg.set(group, []);
            upstreamOrder.get(section).push(group);
        }
        sg.get(group).push(bare);
    }
    const buildSection = (sectionName) => {
        const groups = sectionGroups.get(sectionName);
        if (!groups) {
            return [];
        }
        const result = [];
        const seen = new Set();
        const overlayMap = overlays.get(sectionName) || new Map();
        for (const groupName of sectionOrder.get(sectionName) || []) {
            if (groups.has(groupName)) {
                result.push(buildEntry(groupName, groups.get(groupName), overlayMap.get(groupName)));
                seen.add(groupName);
            }
        }
        for (const groupName of upstreamOrder.get(sectionName)) {
            if (!seen.has(groupName)) {
                result.push(buildEntry(groupName, groups.get(groupName), null));
            }
        }
        return result;
    };
    const buildSubgraph = (prefix) => {
        const sub = {};
        for (const key of ['input', 'blocks', 'output']) {
            const list = buildSection(prefix ? `${prefix}.${key}` : key);
            if (list.length > 0) {
                sub[key] = list;
            }
        }
        return sub;
    };
    const graph = buildSubgraph('');
    const encoder = buildSubgraph('encoder');
    if (Object.keys(encoder).length > 0) {
        graph.encoder = encoder;
    }
    const decoder = buildSubgraph('decoder');
    if (Object.keys(decoder).length > 0) {
        graph.decoder = decoder;
    }
    const result = { name: archName };
    if (existing && existing.family) {
        result.family = existing.family;
    }
    result.graph = graph;
    return result;
};

const stringify = (entries) => {
    const json = JSON.stringify(entries, null, 2);
    let formatted = json.replace(/\[\n\s+("(?:[^"\\]|\\.)*"(?:,\n\s+"(?:[^"\\]|\\.)*")*)\n\s+\]/g,
        (_, inner) => `[${inner.replace(/,\n\s+/g, ', ')}]`);
    formatted = formatted.replace(/\{\n\s+([^{}]*?)\n\s+\}/g,
        (_, inner) => `{ ${inner.replace(/,\n\s+/g, ', ')} }`);
    return `${formatted}\n`;
};

const validate = () => {
    const assert = (actual, expected, label) => {
        const a = JSON.stringify(actual);
        const e = JSON.stringify(expected);
        if (a !== e) {
            throw new Error(`gguf-script self-test ${label}: expected ${e}, got ${a}`);
        }
    };
    const sample = `
MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
    MODEL_ARCH.LLAMA: "llama",
}

TENSOR_NAMES: dict[MODEL_TENSOR, str] = {
    MODEL_TENSOR.TOKEN_EMBD: "token_embd",
    MODEL_TENSOR.OUTPUT_NORM: "output_norm",
    MODEL_TENSOR.OUTPUT: "output",
    MODEL_TENSOR.ATTN_Q: "blk.{bid}.attn_q",
    MODEL_TENSOR.ATTN_K: "blk.{bid}.attn_k",
    MODEL_TENSOR.ATTN_ROT_EMBD: "blk.{bid}.attn_rot_embd",
}

MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
}
`;
    const { archNames, tensorNames, modelTensors } = parseConstants(sample);
    assert(archNames.get('LLAMA'), 'llama', 'parse arch name');
    assert(tensorNames.get('ATTN_Q'), 'blk.{bid}.attn_q', 'parse tensor template');
    assert(modelTensors.get('LLAMA').length, 6, 'parse model tensor count');
    assert(classify('blk.{bid}.attn_q'), { section: 'blocks', bare: 'attn_q' }, 'classify block');
    // Curator's `attn_q` and `attn_k` aliases absorb upstream's matching tensors;
    // `attn_rot_embd` (not aliased) surfaces as its own group.
    const existing = {
        name: 'llama',
        graph: {
            blocks: [{ name: 'attention', type: 'X', tensors: ['attn_q', 'attn_k'] }]
        }
    };
    const out = generate('llama', modelTensors.get('LLAMA'), tensorNames, existing);
    const attention = out.graph.blocks.find((e) => e.name === 'attention');
    assert(attention.tensors, ['attn_q', 'attn_k'], 'curator aliases preserved');
    assert(attention.type, 'X', 'overlay type preserved');
    const bare = out.graph.blocks.find((e) => e.name === 'attn_rot_embd');
    assert(bare !== undefined, true, 'unaliased upstream tensor surfaces as its own group');
};

const metadata = async () => {
    validate();
    const existingText = await fs.readFile(metadataPath, 'utf-8');
    const existingArray = JSON.parse(existingText);
    const text = await fs.readFile(constantsPath, 'utf-8');
    const { archNames, tensorNames, modelTensors } = parseConstants(text);
    const existingByName = new Map(existingArray.map((e) => [e.name, e]));
    const result = [];
    const emitted = new Set();
    const archByName = new Map();
    for (const [enumKey, archName] of archNames) {
        archByName.set(archName, enumKey);
    }
    const emit = (archName) => {
        if (emitted.has(archName) || archName === 'clip') {
            return;
        }
        const existing = existingByName.get(archName);
        // T5-style encoder-decoder archs use a curator-specific convention
        // (duplicated globals, mixed bare/prefixed aliases) that this generator
        // does not model. Preserve them as-is.
        if (existing && existing.graph && (existing.graph.encoder || existing.graph.decoder)) {
            result.push(existing);
            emitted.add(archName);
            return;
        }
        const enumKey = archByName.get(archName);
        if (!enumKey) {
            return;
        }
        const tensorList = modelTensors.get(enumKey);
        if (!tensorList) {
            return;
        }
        result.push(generate(archName, tensorList, tensorNames, existing));
        emitted.add(archName);
    };
    for (const arch of existingArray) {
        emit(arch.name);
    }
    for (const archName of archNames.values()) {
        emit(archName);
    }
    await fs.writeFile(metadataPath, stringify(result));
};

await metadata();
