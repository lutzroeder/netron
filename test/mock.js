
import * as fs from 'fs/promises';
import * as node from '../source/node.js';
import * as path from 'path';
import * as url from 'url';
import * as worker_threads from 'worker_threads';

const mock = {};

mock.Context = class {

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
        const set = new Set(value.split(' ').concat(...tokens));
        this._element.setAttribute('class', Array.from(set).filter((s) => s).join(' '));
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
        return { left: 0, top: 0, width: 0, height: 0 };
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

    requestAnimationFrame(callback) {
        callback();
    }
}

mock.Host = class {

    constructor(environment) {
        this._environment = environment;
        this._errors = [];
        mock.Host.source = mock.Host.source || this._dirname('..', 'source');
        mock.Host.window = mock.Host.window || new Window();
    }

    async view(/* view */) {
    }

    async start() {
    }

    get window() {
        return mock.Host.window;
    }

    get document() {
        return this.window.document;
    }

    get errors() {
        return this._errors;
    }

    get type() {
        return 'Test';
    }

    environment(name) {
        return this._environment[name];
    }

    update() {
    }

    screen(/* name */) {
    }

    async require(id) {
        const file = path.join(mock.Host.source, `${id}.js`);
        return await import(`file://${file}`);
    }

    worker(id) {
        const file = path.join(mock.Host.source, `${id}.js`);
        const worker = new worker_threads.Worker(file);
        worker.addEventListener = (type, listener) => {
            worker.on(type, (message) => listener({ data: message }));
        };
        return worker;
    }

    async request(file, encoding, basename) {
        const pathname = path.join(basename || mock.Host.source, file);
        const exists = await this._access(pathname);
        if (!exists) {
            throw new Error(`The file '${file}' does not exist.`);
        }
        const stats = await fs.stat(pathname);
        if (stats.isDirectory()) {
            throw new Error(`The path '${file}' is a directory.`);
        }
        if (encoding) {
            return await fs.readFile(pathname, encoding);
        }
        return new node.FileStream(pathname, 0, stats.size, stats.mtimeMs);
    }

    event(/* name, params */) {
    }

    exception(error /*, fatal */) {
        this._errors.push(error);
    }

    message() {
    }

    async _access(path) {
        try {
            await fs.access(path);
            return true;
        } catch {
            return false;
        }
    }

    _dirname(...args) {
        const file = url.fileURLToPath(import.meta.url);
        const dir = path.dirname(file);
        return path.join(dir, ...args);
    }
};

export const Host = mock.Host;
export const Context = mock.Context;
