
var host = host || {};

const electron = require('electron');
const fs = require('fs');
const http = require('http');
const https = require('https');
const process = require('process');
const path = require('path');
const querystring = require('querystring');

host.ElectronHost = class {

    constructor() {
        process.on('uncaughtException', (err) => {
            this.exception(err, true);
        });
        this._document = window.document;
        this._window = window;
        this._window.eval = global.eval = () => {
            throw new Error('window.eval() not supported.');
        };
        this._window.addEventListener('unload', () => {
            if (typeof __coverage__ !== 'undefined') {
                const file = path.join('.nyc_output', path.basename(window.location.pathname, '.html')) + '.json';
                /* eslint-disable no-undef */
                fs.writeFileSync(file, JSON.stringify(__coverage__));
                /* eslint-enable no-undef */
            }
        });
        this._environment = electron.ipcRenderer.sendSync('get-environment', {});
        this._queue = [];
    }

    get window() {
        return this._window;
    }

    get document() {
        return this._document;
    }

    get version() {
        return this._environment.version;
    }

    get type() {
        return 'Electron';
    }

    get agent() {
        return 'any';
    }

    initialize(view) {
        this._view = view;
        electron.ipcRenderer.on('open', (_, data) => {
            this._openPath(data.path);
        });
        return new Promise((resolve /*, reject */) => {
            const age = (new Date() - new Date(this._environment.date)) / ( 24 * 60 * 60 * 1000);
            if (age > 180) {
                this._message('Please update to the newest version.', 'Download', () => {
                    const link = this.document.getElementById('logo-github').href;
                    this.openURL(link);
                }, true);
            }
            else {
                const telemetry = () => {
                    if (this._environment.packaged) {
                        this._telemetry = new host.Telemetry('UA-54146-13', this._getConfiguration('userId'), navigator.userAgent, this.type, this.version);
                    }
                    resolve();
                };
                const consent = () => {
                    this._message('This app uses cookies to report errors and anonymous usage information.', 'Accept', () => {
                        this._setConfiguration('consent', Date.now());
                        telemetry();
                    });
                };
                const time = this._getConfiguration('consent');
                if (time && (Date.now() - time) < 30 * 24 * 60 * 60 * 1000) {
                    telemetry();
                }
                else {
                    this._request('https://ipinfo.io/json', { 'Content-Type': 'application/json' }, 2000).then((text) => {
                        try {
                            const json = JSON.parse(text);
                            const countries = ['AT', 'BE', 'BG', 'HR', 'CZ', 'CY', 'DK', 'EE', 'FI', 'FR', 'DE', 'EL', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'NO', 'PL', 'PT', 'SK', 'ES', 'SE', 'GB', 'UK', 'GR', 'EU', 'RO'];
                            if (json && json.country && countries.indexOf(json.country) >= 0) {
                                consent();
                            }
                            else {
                                this._setConfiguration('consent', Date.now());
                                telemetry();
                            }
                        }
                        catch (err) {
                            consent();
                        }
                    }).catch(() => {
                        consent();
                    });
                }
            }
        });
    }

    start() {
        if (this._queue) {
            const queue = this._queue;
            delete this._queue;
            if (queue.length > 0) {
                const path = queue.pop();
                this._openPath(path);
            }
        }

        electron.ipcRenderer.on('export', (_, data) => {
            this._view.export(data.file);
        });
        electron.ipcRenderer.on('cut', () => {
            this._view.cut();
        });
        electron.ipcRenderer.on('copy', () => {
            this._view.copy();
        });
        electron.ipcRenderer.on('paste', () => {
            this._view.paste();
        });
        electron.ipcRenderer.on('selectall', () => {
            this._view.selectAll();
        });
        electron.ipcRenderer.on('toggle', (sender, name) => {
            this._view.toggle(name);
            this._update(Object.assign({}, this._view.options));
        });
        electron.ipcRenderer.on('zoom-in', () => {
            this.document.getElementById('zoom-in-button').click();
        });
        electron.ipcRenderer.on('zoom-out', () => {
            this.document.getElementById('zoom-out-button').click();
        });
        electron.ipcRenderer.on('reset-zoom', () => {
            this._view.resetZoom();
        });
        electron.ipcRenderer.on('show-properties', () => {
            this.document.getElementById('menu-button').click();
        });
        electron.ipcRenderer.on('find', () => {
            this._view.find();
        });
        const menuButton = this.document.getElementById('menu-button');
        if (menuButton) {
            menuButton.setAttribute('title', 'Model Properties');
            menuButton.addEventListener('click', () => {
                this._view.showModelProperties();
            });
        }
        const openFileButton = this.document.getElementById('open-file-button');
        if (openFileButton) {
            openFileButton.style.opacity = 1;
            openFileButton.addEventListener('click', () => {
                electron.ipcRenderer.send('open-file-dialog', {});
            });
        }
        const githubButton = this.document.getElementById('github-button');
        const githubLink = this.document.getElementById('logo-github');
        if (githubButton && githubLink) {
            githubButton.innerText = 'Download';
            githubButton.style.opacity = 1;
            githubButton.addEventListener('click', () => {
                this.openURL(githubLink.href);
            });
        }

        this.document.addEventListener('dragover', (e) => {
            e.preventDefault();
        });
        this.document.addEventListener('drop', (e) => {
            e.preventDefault();
        });
        this.document.body.addEventListener('drop', (e) => {
            e.preventDefault();
            const paths = Array.from(e.dataTransfer.files).map(((file) => file.path));
            if (paths.length > 0) {
                electron.ipcRenderer.send('drop-paths', { paths: paths });
            }
            return false;
        });

        this._view.show('welcome');
    }

    environment(name) {
        return this._environment[name];
    }

    error(message, detail) {
        electron.ipcRenderer.sendSync('show-message-box', {
            type: 'error',
            message: message,
            detail: detail,
        });
    }

    confirm(message, detail) {
        const result = electron.ipcRenderer.sendSync('show-message-box', {
            type: 'question',
            message: message,
            detail: detail,
            buttons: ['Yes', 'No'],
            defaultId: 0,
            cancelId: 1
        });
        return result === 0;
    }

    require(id) {
        try {
            return Promise.resolve(require(id));
        }
        catch (error) {
            return Promise.reject(error);
        }
    }

    save(name, extension, defaultPath, callback) {
        const selectedFile = electron.ipcRenderer.sendSync('show-save-dialog', {
            title: 'Export Tensor',
            defaultPath: defaultPath,
            buttonLabel: 'Export',
            filters: [ { name: name, extensions: [ extension ] } ]
        });
        if (selectedFile) {
            callback(selectedFile);
        }
    }

    export(file, blob) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const data = new Uint8Array(e.target.result);
            fs.writeFile(file, data, null, (err) => {
                if (err) {
                    this.exception(err, false);
                    this.error('Error writing file.', err.message);
                }
            });
        };

        let err = null;
        if (!blob) {
            err = new Error("Export blob is '" + JSON.stringify(blob) + "'.");
        }
        else if (!(blob instanceof Blob)) {
            err = new Error("Export blob type is '" + (typeof blob) + "'.");
        }

        if (err) {
            this.exception(err, false);
            this.error('Error exporting image.', err.message);
        }
        else {
            reader.readAsArrayBuffer(blob);
        }
    }

    request(file, encoding, base) {
        return new Promise((resolve, reject) => {
            const pathname = path.join(base || __dirname, file);
            fs.stat(pathname, (err, stat) => {
                if (err && err.code === 'ENOENT') {
                    reject(new Error("The file '" + file + "' does not exist."));
                }
                else if (err) {
                    reject(err);
                }
                else if (!stat.isFile()) {
                    reject(new Error("The path '" + file + "' is not a file."));
                }
                else if (stat && stat.size < 0x7ffff000) {
                    fs.readFile(pathname, encoding, (err, data) => {
                        if (err) {
                            reject(err);
                        }
                        else {
                            resolve(encoding ? data : new host.ElectronHost.BinaryStream(data));
                        }
                    });
                }
                else if (encoding) {
                    reject(new Error("The file '" + file + "' size (" + stat.size.toString() + ") for encoding '" + encoding + "' is greater than 2 GB."));
                }
                else {
                    resolve(new host.ElectronHost.FileStream(pathname, 0, stat.size, stat.mtimeMs));
                }
            });
        });
    }

    openURL(url) {
        electron.shell.openExternal(url);
    }

    exception(error, fatal) {
        if (this._telemetry && error && error.telemetry !== false) {
            try {
                const name = error.name ? error.name + ': ' : '';
                const message = error.message ? error.message : JSON.stringify(error);
                const description = [ name + message ];
                if (error.stack) {
                    const format = (file, line, column) => {
                        return file.split('\\').join('/').split('/').pop() + ':' + line + ':' + column;
                    };
                    const match = error.stack.match(/\n {4}at (.*) \((.*):(\d*):(\d*)\)/);
                    if (match) {
                        description.push(match[1] + ' (' + format(match[2], match[3], match[4]) + ')');
                    }
                    else {
                        const match = error.stack.match(/\n {4}at (.*):(\d*):(\d*)/);
                        if (match) {
                            description.push('(' + format(match[1], match[2], match[3]) + ')');
                        }
                        else {
                            const match = error.stack.match(/.*\n\s*(.*)\s*/);
                            description.push(match ? match[1] : error.stack.split('\n').shift());
                        }
                    }
                }
                this._telemetry.exception(description.join(' @ '), fatal);
            }
            catch (e) {
                // continue regardless of error
            }
        }
    }

    screen(name) {
        if (this._telemetry) {
            try {
                this._telemetry.screenview(name);
            }
            catch (e) {
                // continue regardless of error
            }
        }
    }

    event(category, action, label, value) {
        if (this._telemetry) {
            try {
                this._telemetry.event(category, action, label, value);
            }
            catch (e) {
                // continue regardless of error
            }
        }
    }

    _context(location) {
        const basename = path.basename(location);
        const stat = fs.statSync(location);
        if (stat.isFile()) {
            const dirname = path.dirname(location);
            return this.request(basename, null, dirname).then((stream) => {
                return new host.ElectronHost.ElectronContext(this, dirname, basename, stream);
            });
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
                        const stream = new host.ElectronHost.FileStream(pathname, 0, stat.size, stat.mtimeMs);
                        const name = pathname.split(path.sep).join(path.posix.sep);
                        entries.set(name, stream);
                    }
                }
            };
            walk(location);
            return Promise.resolve(new host.ElectronHost.ElectronContext(this, location, basename, null, entries));
        }
        throw new Error("Unsupported path stat '" + JSON.stringify(stat) + "'.");
    }

    _openPath(path) {
        if (this._queue) {
            this._queue.push(path);
            return;
        }
        if (path && this._view.accept(path)) {
            this._view.show('welcome spinner');
            this._context(path).then((context) => {
                this._view.open(context).then((model) => {
                    this._view.show(null);
                    const options = Object.assign({}, this._view.options);
                    if (model) {
                        options.path = path;
                    }
                    this._update(options);
                }).catch((error) => {
                    const options = Object.assign({}, this._view.options);
                    if (error) {
                        this._view.error(error, null, null);
                        options.path = null;
                    }
                    this._update(options);
                });
            }).catch((error) => {
                this._view.error(error, 'Error while reading file.', null);
                this._update({ path: null });
            });
        }
    }

    _request(location, headers, timeout) {
        return new Promise((resolve, reject) => {
            const url = new URL(location);
            const protocol = url.protocol === 'https:' ? https : http;
            const options = {};
            options.headers = headers;
            if (timeout) {
                options.timeout = timeout;
            }
            const request = protocol.request(location, options, (response) => {
                if (response.statusCode !== 200) {
                    const err = new Error("The web request failed with status code " + response.statusCode + " at '" + location + "'.");
                    err.type = 'error';
                    err.url = location;
                    err.status = response.statusCode;
                    reject(err);
                }
                else {
                    let data = '';
                    response.on('data', (chunk) => {
                        data += chunk;
                    });
                    response.on('err', (err) => {
                        reject(err);
                    });
                    response.on('end', () => {
                        resolve(data);
                    });
                }
            });
            request.on("error", (err) => {
                reject(err);
            });
            request.on("timeout", () => {
                request.destroy();
                const error = new Error("The web request timed out at '" + location + "'.");
                error.type = 'timeout';
                error.url = url;
                reject(error);
            });
            request.end();
        });
    }

    _getConfiguration(name) {
        return electron.ipcRenderer.sendSync('get-configuration', { name: name });
    }

    _setConfiguration(name, value) {
        electron.ipcRenderer.sendSync('set-configuration', { name: name, value: value });
    }

    _update(data) {
        electron.ipcRenderer.send('update', data);
    }

    _message(message, action, callback, modal) {
        const messageText = this.document.getElementById('message');
        if (messageText) {
            messageText.innerText = message;
        }
        const messageButton = this.document.getElementById('message-button');
        if (messageButton) {
            if (action && callback) {
                messageButton.style.removeProperty('display');
                messageButton.innerText = action;
                messageButton.onclick = () => {
                    if (!modal) {
                        messageButton.onclick = null;
                    }
                    callback();
                };
            }
            else {
                messageButton.style.display = 'none';
                messageButton.onclick = null;
            }
        }
        this._view.show('welcome message');
    }
};

host.Telemetry = class {

    constructor(trackingId, clientId, userAgent, applicationName, applicationVersion) {
        this._params = {
            aip: '1', // anonymizeIp
            tid: trackingId,
            cid: clientId,
            ua: userAgent,
            an: applicationName,
            av: applicationVersion
        };
    }

    screenview(screenName) {
        const params = Object.assign({}, this._params);
        params.cd = screenName;
        this._send('screenview', params);
    }

    event(category, action, label, value) {
        const params = Object.assign({}, this._params);
        params.ec = category;
        params.ea = action;
        params.el = label;
        params.ev = value;
        this._send('event', params);
    }

    exception(description, fatal) {
        const params = Object.assign({}, this._params);
        params.exd = description;
        if (fatal) {
            params.exf = '1';
        }
        this._send('exception', params);
    }

    _send(type, params) {
        params.t = type;
        params.v = '1';
        for (const param in params) {
            if (params[param] === null || params[param] === undefined) {
                delete params[param];
            }
        }
        const body = querystring.stringify(params);
        const options = {
            method: 'POST',
            host: 'www.google-analytics.com',
            path: '/collect',
            headers: { 'Content-Length': Buffer.byteLength(body) }
        };
        const request = https.request(options, (response) => {
            response.on('error', (/* error */) => {});
        });
        request.setTimeout(5000, () => {
            request.destroy();
        });
        request.on('error', (/* error */) => {});
        request.write(body);
        request.end();
    }
};

host.ElectronHost.BinaryStream = class {

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
        return new host.ElectronHost.BinaryStream(buffer.slice(0));
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
};

host.ElectronHost.FileStream = class {

    constructor(file, start, length, mtime) {
        this._file = file;
        this._start = start;
        this._length = length;
        this._position = 0;
        this._mtime = mtime;
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    stream(length) {
        const file = new host.ElectronHost.FileStream(this._file, this._position, length, this._mtime);
        this.skip(length);
        return file;
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._length) {
            throw new Error('Expected ' + (this._position - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    peek(length) {
        length = length !== undefined ? length : this._length - this._position;
        if (length < 0x10000000) {
            const position = this._fill(length);
            this._position -= length;
            return this._buffer.subarray(position, position + length);
        }
        const position = this._position;
        this.skip(length);
        this.seek(position);
        const buffer = new Uint8Array(length);
        this._read(buffer, position);
        return buffer;
    }

    read(length) {
        length = length !== undefined ? length : this._length - this._position;
        if (length < 0x10000000) {
            const position = this._fill(length);
            return this._buffer.subarray(position, position + length);
        }
        const position = this._position;
        this.skip(length);
        const buffer = new Uint8Array(length);
        this._read(buffer, position);
        return buffer;
    }

    byte() {
        const position = this._fill(1);
        return this._buffer[position];
    }

    _fill(length) {
        if (this._position + length > this._length) {
            throw new Error('Expected ' + (this._position + length - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
        if (!this._buffer || this._position < this._offset || this._position + length > this._offset + this._buffer.length) {
            this._offset = this._position;
            this._buffer = new Uint8Array(Math.min(0x10000000, this._length - this._offset));
            this._read(this._buffer, this._offset);
        }
        const position = this._position;
        this._position += length;
        return position - this._offset;
    }

    _read(buffer, offset) {
        const descriptor = fs.openSync(this._file, 'r');
        const stat = fs.statSync(this._file);
        if (stat.mtimeMs != this._mtime) {
            throw new Error("File '" + this._file + "' last modified time changed.");
        }
        try {
            fs.readSync(descriptor, buffer, 0, buffer.length, offset + this._start);
        }
        finally {
            fs.closeSync(descriptor);
        }
    }
};

host.ElectronHost.ElectronContext = class {

    constructor(host, folder, identifier, stream, entries) {
        this._host = host;
        this._folder = folder;
        this._identifier = identifier;
        this._stream = stream;
        this._entries = entries || new Map();
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

window.addEventListener('load', () => {
    global.protobuf = require('./protobuf');
    global.flatbuffers = require('./flatbuffers');
    const view = require('./view');
    window.__host__ = new host.ElectronHost();
    window.__view__ = new view.View(window.__host__);
});
