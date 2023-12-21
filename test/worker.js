
import * as fs from 'fs/promises';
import * as path from 'path';
import * as process from 'process';
import * as url from 'url';
import * as worker_threads from 'worker_threads';
import * as base from '../source/base.js';
import * as view from '../source/view.js';
import * as zip from '../source/zip.js';
import * as tar from '../source/tar.js';

const clearLine = () => {
    if (process.stdout.clearLine) {
        process.stdout.clearLine();
    }
};

const write = (message) => {
    if (process.stdout.write) {
        process.stdout.write(message);
    }
};

const access = async (path) => {
    try {
        await fs.access(path);
        return true;
    } catch (error) {
        return false;
    }
};

const decompress = (buffer) => {
    let archive = zip.Archive.open(buffer, 'gzip');
    if (archive && archive.entries.size == 1) {
        const stream = archive.entries.values().next().value;
        buffer = stream.peek();
    }
    const formats = [ zip, tar ];
    for (const module of formats) {
        archive = module.Archive.open(buffer);
        if (archive) {
            break;
        }
    }
    return archive;
};

const host = {};

host.TestHost = class {

    constructor() {
        this._window = global.window;
        this._document = this._window.document;
        const dirname = path.dirname(url.fileURLToPath(import.meta.url));
        this._sourceDir = path.join(dirname, '..', 'source');
    }

    get window() {
        return this._window;
    }

    get document() {
        return this._document;
    }

    async view(/* view */) {
    }

    async start() {
    }

    environment(name) {
        if (name == 'zoom') {
            return 'none';
        }
        return null;
    }

    screen(/* name */) {
    }

    async require(id) {
        const file = path.join(this._sourceDir, id + '.js');
        return await import('file://' + file);
    }

    async request(file, encoding, basename) {
        const pathname = path.join(basename || this._sourceDir, file);
        const exists = await access(pathname);
        if (!exists) {
            throw new Error("The file '" + file + "' does not exist.");
        }
        if (encoding) {
            const buffer = await fs.readFile(pathname, encoding);
            return buffer;
        }
        const buffer = await fs.readFile(pathname, null);
        return new base.BinaryStream(buffer);
    }

    event(/* name, params */) {
    }

    exception(err /*, fatal */) {
        throw err;
    }
};

host.TestHost.Context = class {

    constructor(host, folder, identifier, stream, entries) {
        this._host = host;
        this._folder = folder;
        this._identifier = identifier;
        this._stream = stream;
        this._entries = entries;
    }

    get identifier() {
        return this._identifier;
    }

    get stream() {
        return this._stream;
    }

    get entries() {
        return this._entries;
    }

    request(file, encoding, base) {
        return this._host.request(file, encoding, base === undefined ? this._folder : base);
    }

    require(id) {
        return this._host.require(id);
    }

    exception(error, fatal) {
        this._host.exception(error, fatal);
    }
};

global.Document = class {

    constructor() {
        this._elements = {};
        this.documentElement = new HTMLElement();
        this.body = new HTMLElement();
    }

    createElement(/* name */) {
        return new HTMLElement();
    }

    createElementNS(/* namespace, name */) {
        return new HTMLElement();
    }

    createTextNode(/* text */) {
        return new HTMLElement();
    }

    getElementById(id) {
        let element = this._elements[id];
        if (!element) {
            element = new HTMLElement();
            this._elements[id] = element;
        }
        return element;
    }

    addEventListener(/* event, callback */) {
    }

    removeEventListener(/* event, callback */) {
    }
};

global.HTMLElement = class {

    constructor() {
        this._childNodes = [];
        this._attributes = new Map();
        this._style = new CSSStyleDeclaration();
    }

    get style() {
        return this._style;

    }

    appendChild(node) {
        this._childNodes.push(node);
    }

    setAttribute(name, value) {
        this._attributes.set(name, value);
    }

    hasAttribute(name) {
        return this._attributes.has(name);
    }

    getAttribute(name) {
        return this._attributes.get(name);
    }

    getElementsByClassName(name) {
        const elements = [];
        for (const node of this._childNodes) {
            if (node instanceof HTMLElement) {
                elements.push(...node.getElementsByClassName(name));
                if (node.hasAttribute('class') &&
                    node.getAttribute('class').split(' ').find((text) => text === name)) {
                    elements.push(node);
                }
            }
        }
        return elements;
    }

    addEventListener(/* event, callback */) {
    }

    removeEventListener(/* event, callback */) {
    }

    get classList() {
        return new DOMTokenList();
    }

    getBBox() {
        return { x: 0, y: 0, width: 10, height: 10 };
    }

    getBoundingClientRect() {
        return { left: 0, top: 0, wigth: 0, height: 0 };
    }

    scrollTo() {
    }

    focus() {
    }
};

global.CSSStyleDeclaration = class {

    constructor() {
        this._properties = new Map();
    }

    setProperty(name, value) {
        this._properties.set(name, value);
    }

    removeProperty(name) {
        this._properties.delete(name);
    }
};

global.DOMTokenList = class {

    add(/* token */) {
    }
};

global.Window = class {

    constructor() {
        this._document = new Document();
    }

    get document() {
        return this._document;
    }

    addEventListener(/* event, callback */) {
    }

    removeEventListener(/* event, callback */) {
    }
};

const request = async (url, init) => {
    const response = await fetch(url, init);
    if (!response.ok) {
        throw new Error(response.status.toString());
    }
    if (response.body) {
        const reader = response.body.getReader();
        const length = response.headers.has('Content-Length') ? Number(response.headers.get('Content-Length')) : -1;
        let position = 0;
        const stream = new ReadableStream({
            start(controller) {
                const read = async () => {
                    try {
                        const result = await reader.read();
                        if (result.done) {
                            clearLine();
                            controller.close();
                        } else {
                            position += result.value.length;
                            if (length >= 0) {
                                const label = url.length > 70 ? url.substring(0, 66) + '...' : url;
                                write('  (' + ('  ' + Math.floor(100 * (position / length))).slice(-3) + '%) ' + label + '\r');
                            } else {
                                write('  ' + position + ' bytes\r');
                            }
                            controller.enqueue(result.value);
                            read();
                        }
                    } catch (error) {
                        controller.error(error);
                    }
                };
                read();
            }
        });
        return new Response(stream, {
            status: response.status,
            statusText: response.statusText,
            headers: response.headers
        });
    }
    return response;
};

export class Target {

    constructor(host, item) {
        Object.assign(this, item);
        this.host = host;
        const target = item.target.split(',');
        this.target = item.type ? target : target.map((target) => path.resolve(process.cwd(), target));
        this.action = new Set((this.action || '').split(';'));
        const dirname = path.dirname(url.fileURLToPath(import.meta.url));
        this.folder = item.type ? path.normalize(path.join(dirname, '..', 'third_party' , 'test', item.type)) : '';
        this.name = this.type ? this.type + '/' + this.target[0] : this.target[0];
        this.measures = new Map([ [ 'name', this.name ] ]);
    }

    static async start() {
        global.window = new Window();
        await zip.Archive.import();
        return new host.TestHost();
    }

    async execute() {
        const time = async (method) => {
            const start = process.hrtime.bigint();
            let err = null;
            try {
                await method.call(this);
            } catch (error) {
                err = error;
            }
            const duration = Number(process.hrtime.bigint() - start) / 1e9;
            this.measures.set(method.name, duration);
            if (err) {
                throw err;
            }
        };
        write(this.name + '\n');
        clearLine();
        await time(this.download);
        try {
            await time(this.load);
            await time(this.validate);
            if (!this.action.has('skip-render')) {
                await time(this.render);
            }
            if (this.error) {
                throw new Error('Expected error.');
            }
        } catch (error) {
            if (!this.error || error.message !== this.error) {
                throw error;
            }
        }
    }

    async download(targets, sources) {
        targets = targets || Array.from(this.target);
        sources = sources || this.source;
        const files = targets.map((file) => path.join(this.folder, file));
        const exists = await Promise.all(files.map((file) => access(file)));
        if (exists.every((value) => value)) {
            return;
        }
        if (!sources) {
            throw new Error('Download source not specified.');
        }
        let source = '';
        let sourceFiles = [];
        const match = sources.match(/^(.*?)\[(.*?)\](.*)$/);
        if (match) {
            [, source, sourceFiles, sources] = match;
            sourceFiles = sourceFiles.split(',').map((file) => file.trim());
            sources = sources && sources.startsWith(',') ? sources.substring(1).trim() : '';
        } else {
            const commaIndex = sources.indexOf(',');
            if (commaIndex != -1) {
                source = sources.substring(0, commaIndex);
                sources = sources.substring(commaIndex + 1);
            } else {
                source = sources;
                sources = '';
            }
        }
        await Promise.all(targets.map((target) => {
            const dir = path.dirname(this.folder + '/' + target);
            return fs.mkdir(dir, { recursive: true });
        }));
        const response = await request(source);
        const buffer = await response.arrayBuffer();
        const data = new Uint8Array(buffer);
        if (sourceFiles.length > 0) {
            clearLine();
            write('  decompress...\r');
            const archive = decompress(data);
            clearLine();
            for (const name of sourceFiles) {
                write('  write ' + name + '\r');
                if (name !== '.') {
                    const stream = archive.entries.get(name);
                    if (!stream) {
                        throw new Error("Entry not found '" + name + '. Archive contains entries: ' + JSON.stringify(Array.from(archive.entries.keys())) + " .");
                    }
                    const target = targets.shift();
                    const buffer = stream.peek();
                    const file = path.join(this.folder, target);
                    /* eslint-disable no-await-in-loop */
                    await fs.writeFile(file, buffer, null);
                    /* eslint-enable no-await-in-loop */
                } else {
                    const target = targets.shift();
                    const dir = path.join(this.folder, target);
                    /* eslint-disable no-await-in-loop */
                    await fs.mkdir(dir, { recursive: true });
                    /* eslint-enable no-await-in-loop */
                }
                clearLine();
            }
        } else {
            const target = targets.shift();
            clearLine();
            write('  write ' + target + '\r');
            await fs.writeFile(this.folder + '/' + target, data, null);
        }
        clearLine();
        if (targets.length > 0 && sources.length > 0) {
            await this.download(targets, sources);
        }
    }

    async load() {
        const target = path.join(this.folder, this.target[0]);
        const identifier = path.basename(target);
        const stat = await fs.stat(target);
        let context = null;
        if (stat.isFile()) {
            const buffer = await fs.readFile(target, null);
            const reader = new base.BinaryStream(buffer);
            const dirname = path.dirname(target);
            context = new host.TestHost.Context(this.host, dirname, identifier, reader, new Map());
        } else if (stat.isDirectory()) {
            const entries = new Map();
            const file = async (pathname) => {
                const buffer = await fs.readFile(pathname, null);
                const stream = new base.BinaryStream(buffer);
                const name = pathname.split(path.sep).join(path.posix.sep);
                entries.set(name, stream);
            };
            const walk = async (dir) => {
                const stats = await fs.readdir(dir, { withFileTypes: true });
                const promises = [];
                for (const stat of stats) {
                    const pathname = path.join(dir, stat.name);
                    if (stat.isDirectory()) {
                        promises.push(walk(pathname));
                    } else if (stat.isFile()) {
                        promises.push(file(pathname));
                    }
                }
                await Promise.all(promises);
            };
            await walk(target);
            context = new host.TestHost.Context(this.host, target, identifier, null, entries);
        }
        const modelFactoryService = new view.ModelFactoryService(this.host);
        this.model = await modelFactoryService.open(context);
    }

    validate() {
        if (!this.model.format || (this.format && this.format != this.model.format)) {
            throw new Error("Invalid model format '" + this.model.format + "'.");
        }
        if (this.producer && this.model.producer != this.producer) {
            throw new Error("Invalid producer '" + this.model.producer + "'.");
        }
        if (this.runtime && this.model.runtime != this.runtime) {
            throw new Error("Invalid runtime '" + this.model.runtime + "'.");
        }
        if (this.assert) {
            for (const assert of this.assert) {
                const parts = assert.split('==').map((item) => item.trim());
                const properties = parts[0].split('.');
                const value = JSON.parse(parts[1].replace(/\s*'|'\s*/g, '"'));
                let context = { model: this.model };
                while (properties.length) {
                    const property = properties.shift();
                    if (context[property] !== undefined) {
                        context = context[property];
                        continue;
                    }
                    const match = /(.*)\[(.*)\]/.exec(property);
                    if (match.length === 3 && context[match[1]] !== undefined) {
                        const array = context[match[1]];
                        const index = parseInt(match[2], 10);
                        if (array[index] !== undefined) {
                            context = array[index];
                            continue;
                        }
                    }
                    throw new Error("Invalid property path: '" + parts[0]);
                }
                if (context !== value) {
                    throw new Error("Invalid '" + context.toString() + "' != '" + assert + "'.");
                }
            }
        }
        if (this.model.version || this.model.description || this.model.author || this.model.license) {
            // continue
        }
        for (const graph of this.model.graphs) {
            const values = new Map();
            const validateValue = (value) => {
                value.name.toString();
                value.name.length;
                value.description;
                value.quantization;
                if (value.type) {
                    value.type.toString();
                }
                if (value.initializer) {
                    value.initializer.type.toString();
                    const tensor = new view.Tensor(value.initializer);
                    if (tensor.encoding !== '<' && tensor.encoding !== '>' && tensor.encoding !== '|') {
                        throw new Error("Tensor encoding '" + tensor.encoding + "' is not implemented.");
                    }
                    if (tensor.layout && (tensor.layout !== 'sparse' && tensor.layout !== 'sparse.coo')) {
                        throw new Error("Tensor layout '" + tensor.layout + "' is not implemented.");
                    }
                    if (!tensor.empty) {
                        if (tensor.type && tensor.type.dataType === '?') {
                            throw new Error('Tensor data type is not defined.');
                        } else if (tensor.type && !tensor.type.shape) {
                            throw new Error('Tensor shape is not defined.');
                        } else {
                            tensor.toString();
                            /*
                            const python = await import('../source/python.js');
                            const tensor = argument.initializer;
                            if (tensor.type && tensor.type.dataType !== '?') {
                                let data_type = tensor.type.dataType;
                                switch (data_type) {
                                    case 'boolean': data_type = 'bool'; break;
                                }
                                const execution = new python.Execution();
                                const bytes = execution.invoke('io.BytesIO', []);
                                const dtype = execution.invoke('numpy.dtype', [ data_type ]);
                                const array = execution.invoke('numpy.asarray', [ tensor.value, dtype ]);
                                execution.invoke('numpy.save', [ bytes, array ]);
                            }
                            */
                        }
                    }
                } else if (value.name.length === 0) {
                    throw new Error('Empty value name.');
                }
                if (value.name.length > 0 && value.initializer === null) {
                    if (!values.has(value.name)) {
                        values.set(value.name, value);
                    } else if (value !== values.get(value.name)) {
                        throw new Error("Duplicate value '" + value.name + "'.");
                    }
                }
            };
            for (const input of graph.inputs) {
                input.name.toString();
                input.name.length;
                for (const value of input.value) {
                    validateValue(value);
                }
            }
            for (const output of graph.outputs) {
                output.name.toString();
                output.name.length;
                for (const value of output.value) {
                    validateValue(value);
                }
            }
            for (const node of graph.nodes) {
                const type = node.type;
                if (!type || typeof type.name != 'string') {
                    throw new Error("Invalid node type '" + JSON.stringify(node.type) + "'.");
                }
                view.Documentation.format(type);
                node.name.toString();
                node.description;
                node.attributes.slice();
                for (const attribute of node.attributes) {
                    attribute.name.toString();
                    attribute.name.length;
                    let value = new view.Formatter(attribute.value, attribute.type).toString();
                    if (value && value.length > 1000) {
                        value = value.substring(0, 1000) + '...';
                    }
                    /* value = */ value.split('<');
                }
                for (const input of node.inputs) {
                    input.name.toString();
                    input.name.length;
                    for (const value of input.value) {
                        validateValue(value);
                    }
                }
                for (const output of node.outputs) {
                    output.name.toString();
                    output.name.length;
                    for (const value of output.value) {
                        validateValue(value);
                    }
                }
                if (node.chain) {
                    for (const chain of node.chain) {
                        chain.name.toString();
                        chain.name.length;
                    }
                }
                // new dialog.NodeSidebar(host, node);
            }
        }
    }

    async render() {
        const current = new view.View(this.host);
        current.options.attributes = true;
        current.options.initializers = true;
        await current.renderGraph(this.model, this.model.graphs[0], current.options);
    }
}

const main = async () => {
    const __host__ = await Target.start();
    worker_threads.parentPort.on('message', async (message) => {
        const data = {};
        try {
            const target = new Target(__host__, message);
            data.name = target.name;
            await target.execute();
            data.measures = target.measures;
        } catch (error) {
            data.error = error.message;
        }
        worker_threads.parentPort.postMessage(data);
    });
};

if (!worker_threads.isMainThread) {
    main();
}
