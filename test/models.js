#!/usr/bin/env node

/* jshint esversion: 6 */
/* eslint "no-console": off */

const fs = require('fs');
const path = require('path');
const process = require('process');
const child_process = require('child_process');
const http = require('http');
const https = require('https');
const util = require('util');
const xmldom = require('xmldom');

const json = require('../source/json');
const protobuf = require('../source/protobuf');
const flatbuffers = require('../source/flatbuffers');
const sidebar = require('../source/view-sidebar.js');
const view = require('../source/view.js');
const zip = require('../source/zip');
const gzip = require('../source/gzip');
const tar = require('../source/tar');
const base = require('../source/base');

global.Int64 = base.Int64;
global.Uint64 = base.Uint64;

global.json = json;
global.protobuf = protobuf;
global.flatbuffers = flatbuffers;

global.DOMParser = xmldom.DOMParser;

global.TextDecoder = class {

    constructor(encoding) {
        if (encoding !== 'ascii') {
            this._decoder = new util.TextDecoder(encoding);
        }
    }

    decode(data) {
        if (this._decoder) {
            return this._decoder.decode(data);
        }

        if (data.length < 32) {
            return String.fromCharCode.apply(null, data);
        }

        const buffer = [];
        let start = 0;
        do {
            let end = start + 32;
            if (end > data.length) {
                end = data.length;
            }
            buffer.push(String.fromCharCode.apply(null, data.subarray(start, end)));
            start = end;
        }
        while (start < data.length);
        return buffer.join('');
    }
};

const filter = process.argv.length > 2 ? new RegExp('^' + process.argv[2].replace(/\./, '\\.').replace(/\*/, '.*')) : null;
const dataFolder = path.normalize(__dirname + '/../third_party/test');
const items = JSON.parse(fs.readFileSync(__dirname + '/models.json', 'utf-8'));

class TestHost {

    constructor() {
        this._document = new HTMLDocument();
    }

    get document() {
        return this._document;
    }

    initialize(/* view */) {
        return Promise.resolve();
    }

    start() {
    }

    environment(name) {
        if (name == 'zoom') {
            return 'none';
        }
        return null;
    }

    screen(/* name */) {
    }

    require(id) {
        try {
            const file = path.join(path.join(__dirname, '../source'), id + '.js');
            return Promise.resolve(require(file));
        }
        catch (error) {
            return Promise.reject(error);
        }
    }

    request(file, encoding, base) {
        const pathname = path.join(base || path.join(__dirname, '../source'), file);
        if (!fs.existsSync(pathname)) {
            return Promise.reject(new Error("The file '" + file + "' does not exist."));
        }
        if (encoding) {
            const text = fs.readFileSync(pathname, encoding);
            return Promise.resolve(text);
        }
        const buffer = fs.readFileSync(pathname, null);
        const stream = new TestBinaryStream(buffer);
        return Promise.resolve(stream);
    }

    event(/* category, action, label, value */) {
    }

    exception(err /*, fatal */) {
        this._raise('exception', { exception: err });
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    _raise(event, data) {
        if (this._events && this._events[event]) {
            for (const callback of this._events[event]) {
                callback(this, data);
            }
        }
    }
}

class TestBinaryStream {

    constructor(buffer) {
        this._buffer = buffer;
        this._length = buffer.length;
        this._position = 0;
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    stream(length) {
        const buffer = this.read(length);
        return new TestBinaryStream(buffer.slice(0));
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
        if (this._position > this._buffer.length) {
            throw new Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    peek(length) {
        if (this._position === 0 && length === undefined) {
            return this._buffer;
        }
        const position = this._position;
        this.skip(length !== undefined ? length : this._length - this._position);
        const end = this._position;
        this.seek(position);
        return this._buffer.subarray(position, end);
    }

    read(length) {
        if (this._position === 0 && length === undefined) {
            this._position = this._length;
            return this._buffer;
        }
        const position = this._position;
        this.skip(length !== undefined ? length : this._length - this._position);
        return this._buffer.subarray(position, this._position);
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._buffer[position];
    }
}

class TestContext {

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
}

class HTMLDocument {

    constructor() {
        this._elements = {};
        this.documentElement = new HTMLHtmlElement();
        this.body = new HTMLBodyElement();
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

class HTMLElement {

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

    getBBox() {
        return { x: 0, y: 0, width: 10, height: 10 };
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
        return new DOMTokenList(this);
    }
}

class HTMLHtmlElement extends HTMLElement {
}

class HTMLBodyElement extends HTMLElement{
}

class CSSStyleDeclaration {

    constructor() {
        this._properties = new Map();
    }

    setProperty(name, value) {
        this._properties.set(name, value);
    }
}

class DOMTokenList {

    add(/* token */) {
    }
}

function makeDir(dir) {
    if (!fs.existsSync(dir)){
        fs.mkdirSync(dir, { recursive: true });
    }
}

function decompress(buffer) {
    let archive = null;
    if (buffer.length >= 18 && buffer[0] === 0x1f && buffer[1] === 0x8b) {
        archive = gzip.Archive.open(buffer);
        if (archive.entries.size == 1) {
            const stream = archive.entries.values().next().value;
            buffer = stream.peek();
        }
    }
    const formats = [ zip, tar ];
    for (const module of formats) {
        archive = module.Archive.open(buffer);
        if (archive) {
            break;
        }
    }
    return archive;
}

function request(location, cookie) {
    const options = { rejectUnauthorized: false };
    let httpRequest = null;
    const url = new URL(location);
    const protocol = url.protocol;
    switch (protocol) {
        case 'http:':
            httpRequest = http.request(location, options);
            break;
        case 'https:':
            httpRequest = https.request(location, options);
            break;
    }
    return new Promise((resolve, reject) => {
        if (!httpRequest) {
            reject(new Error("Unknown HTTP request."));
        }
        if (cookie && cookie.length > 0) {
            httpRequest.setHeader('Cookie', cookie);
        }
        httpRequest.on('response', (response) => {
            resolve(response);
        });
        httpRequest.on('error', (error) => {
            reject(error);
        });
        httpRequest.end();
    });
}

function downloadFile(location, cookie) {
    return request(location, cookie).then((response) => {
        const url = new URL(location);
        if (response.statusCode == 200 &&
            url.hostname == 'drive.google.com' &&
            response.headers['set-cookie'].some((cookie) => cookie.startsWith('download_warning_'))) {
            cookie = response.headers['set-cookie'];
            const download = cookie.filter((cookie) => cookie.startsWith('download_warning_')).shift();
            const confirm = download.split(';').shift().split('=').pop();
            location = location + '&confirm=' + confirm;
            return downloadFile(location, cookie);
        }
        if (response.statusCode == 301 || response.statusCode == 302) {
            if (response.headers.location.startsWith('http://') || response.headers.location.startsWith('https://')) {
                location = response.headers.location;
            }
            else {
                location = url.protocol + '//' + url.hostname + response.headers.location;
            }
            return downloadFile(location, cookie);
        }
        if (response.statusCode != 200) {
            throw new Error(response.statusCode.toString() + ' ' + location);
        }
        return new Promise((resolve, reject) => {
            let position = 0;
            const data = [];
            const length = response.headers['content-length'] ? Number(response.headers['content-length']) : -1;
            response.on('data', (chunk) => {
                position += chunk.length;
                if (length >= 0) {
                    const label = location.length > 70 ? location.substring(0, 66) + '...' : location;
                    process.stdout.write('  (' + ('  ' + Math.floor(100 * (position / length))).slice(-3) + '%) ' + label + '\r');
                }
                else {
                    process.stdout.write('  ' + position + ' bytes\r');
                }
                data.push(chunk);
            });
            response.on('end', () => {
                resolve(Buffer.concat(data));
            });
            response.on('error', (error) => {
                reject(error);
            });
        });
    });
}

function download(folder, targets, sources) {
    if (targets.every((file) => fs.existsSync(folder + '/' + file))) {
        return Promise.resolve();
    }
    if (!sources) {
        return Promise.reject(new Error('Download source not specified.'));
    }
    let source = '';
    let sourceFiles = [];
    const startIndex = sources.indexOf('[');
    const endIndex = sources.indexOf(']');
    if (startIndex != -1 && endIndex != -1 && endIndex > startIndex) {
        sourceFiles = sources.substring(startIndex + 1, endIndex).split(',').map((sourceFile) => sourceFile.trim());
        source = sources.substring(0, startIndex);
        sources = sources.substring(endIndex + 1);
        if (sources.startsWith(',')) {
            sources = sources.substring(1);
        }
    }
    else {
        const commaIndex = sources.indexOf(',');
        if (commaIndex != -1) {
            source = sources.substring(0, commaIndex);
            sources = sources.substring(commaIndex + 1);
        }
        else {
            source = sources;
            sources = '';
        }
    }
    for (const target of targets) {
        makeDir(path.dirname(folder + '/' + target));
    }
    return downloadFile(source).then((data) => {
        if (sourceFiles.length > 0) {
            if (process.stdout.clearLine) {
                process.stdout.clearLine();
            }
            process.stdout.write('  decompress...\r');
            const archive = decompress(data, source.split('?').shift().split('/').pop());
            for (const name of sourceFiles) {
                if (process.stdout.clearLine) {
                    process.stdout.clearLine();
                }
                process.stdout.write('  write ' + name + '\n');
                if (name !== '.') {
                    const stream = archive.entries.get(name);
                    if (!stream) {
                        throw new Error("Entry not found '" + name + '. Archive contains entries: ' + JSON.stringify(archive.entries.map((entry) => entry.name)) + " .");
                    }
                    const target = targets.shift();
                    const buffer = stream.peek();
                    const file = path.join(folder, target);
                    fs.writeFileSync(file, buffer, null);
                }
                else {
                    const target = targets.shift();
                    const dir = path.join(folder, target);
                    if (!fs.existsSync(dir)) {
                        fs.mkdirSync(dir);
                    }
                }
            }
        }
        else {
            const target = targets.shift();
            if (process.stdout.clearLine) {
                process.stdout.clearLine();
            }
            process.stdout.write('  write ' + target + '\r');
            fs.writeFileSync(folder + '/' + target, data, null);
        }
        if (process.stdout.clearLine) {
            process.stdout.clearLine();
        }
        if (sources.length > 0) {
            return download(folder, targets, sources);
        }
        return;
    });
}

function script(folder, targets, command, args) {
    if (targets.every((file) => fs.existsSync(folder + '/' + file))) {
        return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
        try {
            const comspec = process.env.COMSPEC;
            if (process.platform === 'win32' && process.env.SHELL) {
                process.env.COMSPEC = process.env.SHELL;
                command = '/' + command.split(':').join('').split('\\').join('/');
            }
            child_process.execSync(command + ' ' + args, { stdio: [ 0, 1 , 2] });
            process.env.COMSPEC = comspec;
            resolve();
        }
        catch (error) {
            reject(error);
        }
    });
}

function loadModel(target, item) {
    const host = new TestHost();
    const exceptions = [];
    host.on('exception', (_, data) => {
        exceptions.push(data.exception);
    });
    const identifier = path.basename(target);
    const stat = fs.statSync(target);
    let context = null;
    if (stat.isFile()) {
        const buffer = fs.readFileSync(target, null);
        const reader = new TestBinaryStream(buffer);
        const dirname = path.dirname(target);
        context = new TestContext(host, dirname, identifier, reader);
    }
    else if (stat.isDirectory()) {
        const entries = new Map();
        const walk = (dir) => {
            for (const item of fs.readdirSync(dir)) {
                const pathname = path.join(dir, item);
                const stat = fs.statSync(pathname);
                if (stat.isDirectory()) {
                    walk(pathname);
                }
                else if (stat.isFile()) {
                    const buffer = fs.readFileSync(pathname, null);
                    const stream = new TestBinaryStream(buffer);
                    const name = pathname.split(path.sep).join(path.posix.sep);
                    entries.set(name, stream);
                }
            }
        };
        walk(target);
        context = new TestContext(host, target, identifier, null, entries);
    }
    const modelFactoryService = new view.ModelFactoryService(host);
    let opened = false;
    return modelFactoryService.open(context).then((model) => {
        if (opened) {
            throw new Error("Model opened more than once '" + target + "'.");
        }
        opened = true;
        if (!model.format || (item.format && model.format != item.format)) {
            throw new Error("Invalid model format '" + model.format + "'.");
        }
        if (item.producer && model.producer != item.producer) {
            throw new Error("Invalid producer '" + model.producer + "'.");
        }
        if (item.runtime && model.runtime != item.runtime) {
            throw new Error("Invalid runtime '" + model.runtime + "'.");
        }
        if (item.assert) {
            for (const assert of item.assert) {
                const parts = assert.split('=').map((item) => item.trim());
                const properties = parts[0].split('.');
                const value = parts[1];
                let context = { model: model };
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
                if (context !== value.toString()) {
                    throw new Error("Invalid '" + value.toString() + "' != '" + assert + "'.");
                }
            }
        }
        model.version;
        model.description;
        model.author;
        model.license;
        for (const graph of model.graphs) {
            for (const input of graph.inputs) {
                input.name.toString();
                input.name.length;
                for (const argument of input.arguments) {
                    argument.name.toString();
                    argument.name.length;
                    if (argument.type) {
                        argument.type.toString();
                    }
                }
            }
            for (const output of graph.outputs) {
                output.name.toString();
                output.name.length;
                for (const argument of output.arguments) {
                    argument.name.toString();
                    argument.name.length;
                    if (argument.type) {
                        argument.type.toString();
                    }
                }
            }
            for (const node of graph.nodes) {
                node.type.toString();
                node.type.length;
                if (!node.type || typeof node.type.name != 'string') {
                    throw new Error("Invalid node type '" + JSON.stringify(node.type) + "'.");
                }
                sidebar.DocumentationSidebar.formatDocumentation(node.type);
                node.name.toString();
                node.name.length;
                node.description;
                node.attributes.slice();
                for (const attribute of node.attributes) {
                    attribute.name.toString();
                    attribute.name.length;
                    let value = sidebar.NodeSidebar.formatAttributeValue(attribute.value, attribute.type);
                    if (value && value.length > 1000) {
                        value = value.substring(0, 1000) + '...';
                    }
                    value = value.split('<');
                }
                for (const input of node.inputs) {
                    input.name.toString();
                    input.name.length;
                    for (const argument of input.arguments) {
                        argument.name.toString();
                        argument.name.length;
                        argument.description;
                        if (argument.type) {
                            argument.type.toString();
                        }
                        if (argument.initializer) {
                            argument.initializer.toString();
                            argument.initializer.type.toString();
                        }
                    }
                }
                for (const output of node.outputs) {
                    output.name.toString();
                    output.name.length;
                    for (const argument of output.arguments) {
                        argument.name.toString();
                        argument.name.length;
                        if (argument.type) {
                            argument.type.toString();
                        }
                    }
                }
                if (node.chain) {
                    for (const chain of node.chain) {
                        chain.name.toString();
                        chain.name.length;
                    }
                }
                // new sidebar.NodeSidebar(host, node);
            }
        }
        if (exceptions.length > 0) {
            throw exceptions[0];
        }
        return model;
    });
}

function render(model) {
    try {
        const host = new TestHost();
        const currentView = new view.View(host);
        if (!currentView.showAttributes) {
            currentView.toggleAttributes();
        }
        if (!currentView.showInitializers) {
            currentView.toggleInitializers();
        }
        return currentView.renderGraph(model, model.graphs[0]);
    }
    catch (error) {
        return Promise.reject(error);
    }
}

function next() {
    if (items.length == 0) {
        return;
    }
    const item = items.shift();
    if (!item.type) {
        console.error("Property 'type' is required for item '" + JSON.stringify(item) + "'.");
        return;
    }
    const targets = item.target.split(',');
    const target = targets[0];
    const folder = dataFolder + '/' + item.type;
    const name = item.type + '/' + target;
    if (filter && !filter.test(name)) {
        next();
        return;
    }
    process.stdout.write(item.type + '/' + target + '\n');
    if (item.action && item.action.split(';').some((action) => action == 'skip')) {
        next();
        return;
    }
    if (process.stdout.clearLine) {
        process.stdout.clearLine();
    }

    let promise = null;
    if (item.script) {
        const index = item.script.search(' ');
        const root = path.dirname(__dirname);
        const command = path.resolve(root, item.script.substring(0, index));
        const args = item.script.substring(index + 1);
        promise = script(folder, targets, command, args);
    }
    else {
        const sources = item.source;
        promise = download(folder, targets, sources);
    }
    return promise.then(() => {
        return loadModel(folder + '/' + target, item).then((model) => {
            let promise = null;
            if (item.action && item.action.split(';').some((action) => action == 'skip-render')) {
                promise = Promise.resolve();
            }
            else {
                promise = render(model);
            }
            return promise.then(() => {
                if (item.error) {
                    console.error('Expected error.');
                }
                else {
                    return next();
                }
            });
        });
    }).catch((error) => {
        if (!item.error || item.error != error.message) {
            console.error(error.message);
        }
        else {
            return next();
        }
    });
}

next();
