
import * as base from '../source/base.js';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as process from 'process';
import * as python from '../source/python.js';
import * as tar from '../source/tar.js';
import * as url from 'url';
import * as view from '../source/view.js';
import * as worker_threads from 'worker_threads';
import * as zip from '../source/zip.js';

const access = async (path) => {
    try {
        await fs.access(path);
        return true;
    } catch {
        return false;
    }
};

const dirname = (...args) => {
    const file = url.fileURLToPath(import.meta.url);
    const dir = path.dirname(file);
    return path.join(dir, ...args);
};

const decompress = (buffer) => {
    let archive = zip.Archive.open(buffer, 'gzip');
    if (archive && archive.entries.size === 1) {
        const stream = archive.entries.values().next().value;
        buffer = stream.peek();
    }
    const formats = [zip, tar];
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

    constructor(window, environment) {
        this._window = window;
        this._environment = environment;
        this._errors = [];
        host.TestHost.source = host.TestHost.source || dirname('..', 'source');
    }

    async view(/* view */) {
    }

    async start() {
    }

    get window() {
        return this._window;
    }

    get document() {
        return this._window.document;
    }

    get errors() {
        return this._errors;
    }

    environment(name) {
        return this._environment[name];
    }

    screen(/* name */) {
    }

    async require(id) {
        const file = path.join(host.TestHost.source, `${id}.js`);
        return await import(`file://${file}`);
    }

    worker(id) {
        const file = path.join(host.TestHost.source, `${id}.js`);
        const worker = new worker_threads.Worker(file);
        worker.addEventListener = (type, listener) => {
            worker.on(type, (message) => listener({ data: message }));
        };
        return worker;
    }

    async request(file, encoding, basename) {
        const pathname = path.join(basename || host.TestHost.source, file);
        const exists = await access(pathname);
        if (!exists) {
            throw new Error(`The file '${file}' does not exist.`);
        }
        if (encoding) {
            return await fs.readFile(pathname, encoding);
        }
        const buffer = await fs.readFile(pathname, null);
        return new base.BinaryStream(buffer);
    }

    event(/* name, params */) {
    }

    exception(error /*, fatal */) {
        this._errors.push(error);
    }

    message() {
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

    async request(file, encoding, base) {
        return this._host.request(file, encoding, base === undefined ? this._folder : base);
    }

    async require(id) {
        return this._host.require(id);
    }

    error(error, fatal) {
        this._host.exception(error, fatal);
    }
};

class CSSStyleDeclaration {

    constructor() {
        this._properties = new Map();
    }

    setProperty(name, value) {
        this._properties.set(name, value);
    }

    removeProperty(name) {
        this._properties.delete(name);
    }
}

class DOMTokenList {

    constructor(element) {
        this._element = element;
    }

    add(...tokens) {
        const value = this._element.getAttribute('class') || '';
        tokens = new Set(value.split(' ').concat(tokens));
        this._element.setAttribute('class', Array.from(tokens).join(' '));
    }

    contains(token) {
        const value = this._element.getAttribute('class');
        if (value === null || value.indexOf(token) === -1) {
            return false;
        }
        return value.split(' ').some((s) => s === token);
    }
}

class HTMLElement {

    constructor() {
        this._childNodes = [];
        this._attributes = new Map();
        this._style = new CSSStyleDeclaration();
    }

    get style() {
        return this._style;

    }

    get childNodes() {
        return this._childNodes;
    }

    get firstChild() {
        return this._childNodes.length > 0 ? this._childNodes[0] : null;
    }

    get lastChild() {
        const index = this._childNodes.length - 1;
        if (index >= 0) {
            return this._childNodes[index];
        }
        return null;
    }

    appendChild(node) {
        this._childNodes.push(node);
    }

    insertBefore(newNode, referenceNode) {
        const index = this._childNodes.indexOf(referenceNode);
        if (index !== -1) {
            this._childNodes.splice(index, 0, newNode);
        }
    }

    removeChild(node) {
        const index = this._childNodes.lastIndexOf(node);
        if (index !== -1) {
            this._childNodes.splice(index, 1);
        }
    }

    setAttribute(name, value) {
        this._attributes.set(name, value);
    }

    hasAttribute(name) {
        return this._attributes.has(name);
    }

    getAttribute(name) {
        return this._attributes.has(name) ? this._attributes.get(name) : null;
    }

    getElementsByClassName(name) {
        const elements = [];
        for (const node of this._childNodes) {
            if (node instanceof HTMLElement) {
                elements.push(...node.getElementsByClassName(name));
                if (node.classList.contains(name)) {
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
        this._classList = this._classList || new DOMTokenList(this);
        return this._classList;
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
}

class Document {

    constructor() {
        this._elements = {};
        this._documentElement = new HTMLElement();
        this._body = new HTMLElement();
    }

    get documentElement() {
        return this._documentElement;
    }

    get body() {
        return this._body;
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
}

class Window {

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
}

export class Target {

    constructor(item) {
        Object.assign(this, item);
        this.events = {};
        this.tags = new Set(this.tags);
        this.folder = item.type ? path.normalize(dirname('..', 'third_party' , 'test', item.type)) : process.cwd();
        this.assert = !this.assert || Array.isArray(this.assert) ? this.assert : [this.assert];
    }

    on(event, callback) {
        this.events[event] = this.events[event] || [];
        this.events[event].push(callback);
    }

    emit(event, data) {
        if (this.events && this.events[event]) {
            for (const callback of this.events[event]) {
                callback(this, data);
            }
        }
    }

    status(message) {
        this.emit('status', message);
    }

    async execute() {
        if (this.measures) {
            this.measures.set('name', this.name);
        }
        await zip.Archive.import();
        this.window = this.window || new Window();
        const environment = {
            zoom: 'none',
            measure: this.measures ? true : false
        };
        this.host = await new host.TestHost(this.window, environment);
        this.view = new view.View(this.host);
        this.view.options.attributes = true;
        this.view.options.initializers = true;
        const time = async (method) => {
            const start = process.hrtime.bigint();
            let err = null;
            try {
                await method.call(this);
            } catch (error) {
                err = error;
            }
            const duration = Number(process.hrtime.bigint() - start) / 1e9;
            if (this.measures) {
                this.measures.set(method.name, duration);
            }
            if (err) {
                throw err;
            }
        };
        this.status({ name: 'name', target: this.name });
        const errors = [];
        try {
            await time(this.download);
            await time(this.load);
            await time(this.validate);
            if (!this.tags.has('skip-render')) {
                await time(this.render);
            }
        } catch (error) {
            errors.push(error);
        }
        errors.push(...this.host.errors);
        if (errors.length === 0 && this.error) {
            throw new Error('Expected error.');
        }
        if (errors.length > 0 && (!this.error || errors.map((error) => error.message).join('\n') !== this.error)) {
            throw errors[0];
        }
    }

    async request(url, init) {
        const response = await fetch(url, init);
        if (!response.ok) {
            throw new Error(response.status.toString());
        }
        if (response.body) {
            const reader = response.body.getReader();
            const length = response.headers.has('Content-Length') ? parseInt(response.headers.get('Content-Length'), 10) : -1;
            let position = 0;
            const target = this;
            const stream = new ReadableStream({
                async start(controller) {
                    const read = async () => {
                        try {
                            const result = await reader.read();
                            if (result.done) {
                                target.status({ name: 'download' });
                                controller.close();
                            } else {
                                position += result.value.length;
                                if (length >= 0) {
                                    const percent = position / length;
                                    target.status({ name: 'download', target: url, percent });
                                } else {
                                    target.status({ name: 'download', target: url, position });
                                }
                                controller.enqueue(result.value);
                                return await read();
                            }
                        } catch (error) {
                            controller.error(error);
                            throw error;
                        }

                        return null;
                    };
                    return read();
                }
            });
            return new Response(stream, {
                status: response.status,
                statusText: response.statusText,
                headers: response.headers
            });
        }
        return response;
    }
    async download(targets, sources) {
        targets = targets || Array.from(this.targets);
        sources = sources || this.source;
        const files = targets.map((file) => path.resolve(this.folder, file));
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
            if (commaIndex === -1) {
                source = sources;
                sources = '';
            } else {
                source = sources.substring(0, commaIndex);
                sources = sources.substring(commaIndex + 1);
            }
        }
        await Promise.all(targets.map((target) => {
            const dir = path.dirname(`${this.folder}/${target}`);
            return fs.mkdir(dir, { recursive: true });
        }));
        const response = await this.request(source);
        const buffer = await response.arrayBuffer();
        const data = new Uint8Array(buffer);
        if (sourceFiles.length > 0) {
            this.status({ name: 'decompress' });
            const archive = decompress(data);
            for (const name of sourceFiles) {
                this.status({ name: 'write', target: name });
                if (name === '.') {
                    const target = targets.shift();
                    const dir = path.join(this.folder, target);
                    /* eslint-disable no-await-in-loop */
                    await fs.mkdir(dir, { recursive: true });
                    /* eslint-enable no-await-in-loop */
                } else {
                    const stream = archive.entries.get(name);
                    if (!stream) {
                        throw new Error(`Entry not found '${name}. Archive contains entries: ${JSON.stringify(Array.from(archive.entries.keys()))} .`);
                    }
                    const target = targets.shift();
                    const buffer = stream.peek();
                    const file = path.join(this.folder, target);
                    /* eslint-disable no-await-in-loop */
                    await fs.writeFile(file, buffer, null);
                    /* eslint-enable no-await-in-loop */
                }
            }
        } else {
            const target = targets.shift();
            this.status({ name: 'write', target });
            await fs.writeFile(`${this.folder}/${target}`, data, null);
        }
        if (targets.length > 0 && sources.length > 0) {
            await this.download(targets, sources);
        }
    }

    async load() {
        const target = path.resolve(this.folder, this.targets[0]);
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

    async validate() {
        if (!this.model.format || (this.format && this.format !== this.model.format)) {
            throw new Error(`Invalid model format '${this.model.format}'.`);
        }
        if (this.producer && this.model.producer !== this.producer) {
            throw new Error(`Invalid producer '${this.model.producer}'.`);
        }
        if (this.runtime && this.model.runtime !== this.runtime) {
            throw new Error(`Invalid runtime '${this.model.runtime}'.`);
        }
        if (this.model.metadata && !Array.isArray(this.model.metadata) && this.model.metadata.every((argument) => argument.name && argument.value)) {
            throw new Error("Invalid model metadata.'");
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
                    if (match && match.length === 3 && context[match[1]] !== undefined) {
                        const array = context[match[1]];
                        const index = parseInt(match[2], 10);
                        if (array[index] !== undefined) {
                            context = array[index];
                            continue;
                        }
                    }
                    throw new Error(`Invalid property path: '${parts[0]}`);
                }
                if (context !== value) {
                    throw new Error(`Invalid '${context}' != '${assert}'.`);
                }
            }
        }
        if (this.model.version || this.model.description || this.model.author || this.model.license) {
            // continue
        }
        /* eslint-disable no-unused-expressions */
        const validateGraph = async (graph) => {
            const values = new Map();
            const validateValue = async (value) => {
                if (value === null) {
                    return;
                }
                value.name.toString();
                value.name.length;
                value.description;
                if (value.quantization) {
                    if (!this.tags.has('quantization')) {
                        throw new Error("Invalid 'quantization' tag.");
                    }
                    const quantization = new view.Quantization(value.quantization);
                    quantization.toString();
                }
                if (value.type) {
                    value.type.toString();
                }
                if (value.initializer) {
                    value.initializer.type.toString();
                    if (value.initializer && value.initializer.peek && !value.initializer.peek()) {
                        await value.initializer.read();
                    }
                    const tensor = new base.Tensor(value.initializer);
                    if (!this.tags.has('skip-tensor-value')) {
                        if (tensor.encoding !== '<' && tensor.encoding !== '>' && tensor.encoding !== '|') {
                            throw new Error(`Tensor encoding '${tensor.encoding}' is not implemented.`);
                        }
                        if (tensor.layout && (tensor.layout !== 'sparse' && tensor.layout !== 'sparse.coo')) {
                            throw new Error(`Tensor layout '${tensor.layout}' is not implemented.`);
                        }
                        if (!tensor.empty) {
                            if (tensor.type && tensor.type.dataType === '?') {
                                throw new Error('Tensor data type is not defined.');
                            } else if (tensor.type && !tensor.type.shape) {
                                throw new Error('Tensor shape is not defined.');
                            } else {
                                tensor.toString();
                                if (this.tags.has('validation')) {
                                    const size = tensor.type.shape.dimensions.reduce((a, b) => a * b, 1);
                                    if (tensor.type && tensor.type.dataType !== '?' && size < 8192) {
                                        let data_type = '?';
                                        switch (tensor.type.dataType) {
                                            case 'boolean': data_type = 'bool'; break;
                                            case 'bfloat16': data_type = 'float32'; break;
                                            case 'float8e5m2': data_type = 'float16'; break;
                                            case 'float8e5m2fnuz': data_type = 'float16'; break;
                                            case 'float8e4m3fn': data_type = 'float16'; break;
                                            case 'float8e4m3fnuz': data_type = 'float16'; break;
                                            case 'int4': data_type = 'int8'; break;
                                            default: data_type = tensor.type.dataType; break;
                                        }
                                        Target.execution = Target.execution || new python.Execution();
                                        const execution = Target.execution;
                                        const bytes = execution.invoke('io.BytesIO', []);
                                        const dtype = execution.invoke('numpy.dtype', [data_type]);
                                        const array = execution.invoke('numpy.asarray', [tensor.value, dtype]);
                                        execution.invoke('numpy.save', [bytes, array]);
                                    }
                                }
                            }
                        }
                    }
                } else if (value.name.length === 0) {
                    throw new Error('Empty value name.');
                }
                if (value.name.length > 0 && value.initializer === null) {
                    if (!values.has(value.name)) {
                        values.set(value.name, value);
                    } else if (value !== values.get(value.name)) {
                        throw new Error(`Duplicate value '${value.name}'.`);
                    }
                }
            };
            const signatures = Array.isArray(graph.signatures) ? graph.signatures : [graph];
            for (const signature of signatures) {
                for (const input of signature.inputs) {
                    input.name.toString();
                    input.name.length;
                    for (const value of input.value) {
                        /* eslint-disable no-await-in-loop */
                        await validateValue(value);
                        /* eslint-enable no-await-in-loop */
                    }
                }
                for (const output of signature.outputs) {
                    output.name.toString();
                    output.name.length;
                    for (const value of output.value) {
                        /* eslint-disable no-await-in-loop */
                        await validateValue(value);
                        /* eslint-enable no-await-in-loop */
                    }
                }
            }
            if (graph.metadata && !Array.isArray(graph.metadata) && graph.metadata.every((argument) => argument.name && argument.value)) {
                throw new Error("Invalid graph metadata.'");
            }
            for (const node of graph.nodes) {
                const type = node.type;
                if (!type || typeof type.name !== 'string') {
                    throw new Error(`Invalid node type '${JSON.stringify(node.type)}'.`);
                }
                if (Array.isArray(type.nodes)) {
                    /* eslint-disable no-await-in-loop */
                    await validateGraph(type);
                    /* eslint-enable no-await-in-loop */
                }
                view.Documentation.open(type);
                node.name.toString();
                node.description;
                if (node.metadata && !Array.isArray(node.metadata) && node.metadata.every((argument) => argument.name && argument.value)) {
                    throw new Error("Invalid graph metadata.'");
                }
                const attributes = node.attributes;
                if (attributes) {
                    for (const attribute of attributes) {
                        attribute.name.toString();
                        attribute.name.length;
                        const type = attribute.type;
                        const value = attribute.value;
                        if ((type === 'graph' || type === 'function') && value && Array.isArray(value.nodes)) {
                            /* eslint-disable no-await-in-loop */
                            await validateGraph(value);
                            /* eslint-enable no-await-in-loop */
                        } else {
                            let text = new view.Formatter(attribute.value, attribute.type).toString();
                            if (text && text.length > 1000) {
                                text = `${text.substring(0, 1000)}...`;
                            }
                            /* value = */ text.split('<');
                        }
                    }
                }
                const inputs = node.inputs;
                if (Array.isArray(inputs)) {
                    for (const input of inputs) {
                        input.name.toString();
                        input.name.length;
                        if (!input.type || input.type.endsWith('*')) {
                            for (const value of input.value) {
                                /* eslint-disable no-await-in-loop */
                                await validateValue(value);
                                /* eslint-enable no-await-in-loop */
                            }
                            if (this.tags.has('validation')) {
                                if (input.value.length === 1 && input.value[0].initializer) {
                                    const sidebar = new view.TensorSidebar(this.view, input);
                                    sidebar.render();
                                }
                            }
                        }
                    }
                }
                const outputs = node.outputs;
                if (Array.isArray(outputs)) {
                    for (const output of node.outputs) {
                        output.name.toString();
                        output.name.length;
                        if (!output.type || output.type.endsWith('*')) {
                            for (const value of output.value) {
                                /* eslint-disable no-await-in-loop */
                                await validateValue(value);
                                /* eslint-enable no-await-in-loop */
                            }
                        }
                    }
                }
                if (node.chain) {
                    for (const chain of node.chain) {
                        chain.name.toString();
                        chain.name.length;
                    }
                }
                const sidebar = new view.NodeSidebar(this.view, node);
                sidebar.render();
            }
            const sidebar = new view.ModelSidebar(this.view, this.model, graph);
            sidebar.render();
        };
        /* eslint-enable no-unused-expressions */
        for (const graph of this.model.graphs) {
            /* eslint-disable no-await-in-loop */
            await validateGraph(graph);
            /* eslint-enable no-await-in-loop */
        }
    }

    async render() {
        for (const graph of this.model.graphs) {
            const signatures = Array.isArray(graph.signatures) && graph.signatures.length > 0 ? graph.signatures : [graph];
            for (const signature of signatures) {
                /* eslint-disable no-await-in-loop */
                await this.view.renderGraph(this.model, graph, signature, this.view.options);
                /* eslint-enable no-await-in-loop */
            }
        }
    }
}

if (!worker_threads.isMainThread) {
    worker_threads.parentPort.addEventListener('message', async (e) => {
        const message = e.data;
        const response = {};
        try {
            const target = new Target(message);
            response.type = 'complete';
            response.target = target.name;
            target.on('status', (_, message) => {
                message = { type: 'status', ...message };
                worker_threads.parentPort.postMessage(message);
            });
            if (message.measures) {
                target.measures = new Map();
            }
            await target.execute();
            response.measures = target.measures;
        } catch (error) {
            response.type = 'error';
            response.error = {
                name: error.name,
                message: error.message
            };
            const cause = error.cause;
            if (cause) {
                response.error.cause = {
                    name: cause.name,
                    message: cause.message
                };
            }
        }
        worker_threads.parentPort.postMessage(response);
    });
}
