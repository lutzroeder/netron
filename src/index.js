/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */
/* eslint "no-global-assign": ["error", {"exceptions": [ "TextDecoder", "TextEncoder" ] } ] */
/* global view */

var host = {};

host.BrowserHost = class {

    constructor() {
        if (!window.ga) {
            window.GoogleAnalyticsObject = 'ga';
            window.ga = window.ga || function() {
                window.ga.q = window.ga.q || [];
                window.ga.q.push(arguments);
            };
            window.ga.l = 1 * new Date();
        }
        window.ga('create', 'UA-54146-13', 'auto');

        window.addEventListener('error', (e) => {
            this.exception(e.error, true);
        });
        window.eval = () => {
            throw new Error('window.eval() not supported.');
        };
    }

    get document() {
        return window.document;
    }

    get version() {
        return this._version;
    }

    get type() {
        return this._type;
    }

    initialize(view) {
        this._view = view;

        var meta = {};
        for (var element of Array.from(document.getElementsByTagName('meta'))) {
            if (element.content) {
                meta[element.name] = meta[element.name] || [];
                meta[element.name].push(element.content);
            }
        }

        this._version = meta.version ? meta.version[0] : null;
        this._type = meta.type ? meta.type[0] : 'Browser';

        this._zoom = this._getQueryParameter('zoom') || 'd3';

        this._menu = new host.Dropdown(this.document, 'menu-button', 'menu-dropdown');
        this._menu.add({
            label: 'Properties...',
            accelerator: 'CmdOrCtrl+Enter',
            click: () => this._view.showModelProperties()
        });
        this._menu.add({});
        this._menu.add({
            label: 'Find...',
            accelerator: 'CmdOrCtrl+F',
            click: () => this._view.find()
        });
        this._menu.add({});
        this._menu.add({
            label: () => this._view.showAttributes ? 'Hide Attributes' : 'Show Attributes',
            accelerator: 'CmdOrCtrl+D',
            click: () => this._view.toggleAttributes()
        });
        this._menu.add({
            label: () => this._view.showInitializers ? 'Hide Initializers' : 'Show Initializers',
            accelerator: 'CmdOrCtrl+I',
            click: () => this._view.toggleInitializers()
        });
        this._menu.add({
            label: () => this._view.showNames ? 'Hide Names' : 'Show Names',
            accelerator: 'CmdOrCtrl+U',
            click: () => this._view.toggleNames()
        });
        this._menu.add({});
        this._menu.add({
            label: 'Zoom In',
            accelerator: 'Shift+Up',
            click: () => this.document.getElementById('zoom-in-button').click()
        });
        this._menu.add({
            label: 'Zoom Out',
            accelerator: 'Shift+Down',
            click: () => this.document.getElementById('zoom-out-button').click()
        });
        this._menu.add({
            label: 'Actual Size',
            accelerator: 'Shift+Backspace',
            click: () => this._view.resetZoom()
        });
        this._menu.add({});
        this._menu.add({
            label: 'Export as PNG',
            accelerator: 'CmdOrCtrl+Shift+E',
            click: () => this._view.export(document.title + '.png')
        });
        this._menu.add({
            label: 'Export as SVG',
            accelerator: 'CmdOrCtrl+Alt+E',
            click: () => this._view.export(document.title + '.svg')
        });
        this.document.getElementById('menu-button').addEventListener('click', (e) => {
            this._menu.toggle();
            e.preventDefault();
        });

        if (meta.file) {
            this._openModel(meta.file[0], null);
            return;
        }

        var urlParam = this._getQueryParameter('url');
        if (urlParam) {
            this._openModel(urlParam, this._getQueryParameter('identifier') || null);
            return;
        }

        var gistParam = this._getQueryParameter('gist');
        if (gistParam) {
            this._openGist(gistParam);
            return;
        }

        this._view.show('Welcome');
        var openFileButton = this.document.getElementById('open-file-button');
        var openFileDialog = this.document.getElementById('open-file-dialog');
        if (openFileButton && openFileDialog) {
            openFileButton.addEventListener('click', () => {
                openFileDialog.value = '';
                openFileDialog.click();
            });
            openFileDialog.addEventListener('change', (e) => {
                if (e.target && e.target.files && e.target.files.length > 0) {
                    var files = Array.from(e.target.files);
                    var file = files.find((file) => this._view.accept(file.name));
                    if (file) {
                        this._open(file, files);
                    }
                }
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
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                var files = Array.from(e.dataTransfer.files);
                var file = files.find((file) => this._view.accept(file.name));
                if (file) {
                    this._open(file, files);
                }
            }
        });
    }

    environment(name) {
        if (name == 'zoom') {
            return this._zoom;
        }
        return null;
    }

    error(message, detail) {
        alert((message == 'Error' ? '' : message + ' ') + detail);
    }

    confirm(message, detail) {
        return confirm(message + ' ' + detail);
    }

    require(id) {
        var url = this._url(id + '.js');
        window.__modules__ = window.__modules__ || {};
        if (window.__modules__[url]) {
            return Promise.resolve(window.__exports__[url]);
        }
        return new Promise((resolve, reject) => {
            window.module = { exports: {} };
            var script = document.createElement('script');
            script.setAttribute('id', id);
            script.setAttribute('type', 'text/javascript');
            script.setAttribute('src', url);
            script.onload = () => {
                var exports = window.module.exports;
                delete window.module;
                window.__modules__[id] = exports;
                resolve(exports);
            };
            script.onerror = (e) => {
                delete window.module;
                reject(new Error('The script \'' + e.target.src + '\' failed to load.'));
            };
            this.document.head.appendChild(script);
        });
    }

    save(name, extension, defaultPath, callback) {
        callback(defaultPath + '.' + extension);
    }

    export(file, blob) {
        var element = this.document.createElement('a');
        element.download = file;
        element.href = URL.createObjectURL(blob);
        this.document.body.appendChild(element);
        element.click();
        this.document.body.removeChild(element);
    }

    request(base, file, encoding) {
        return new Promise((resolve, reject) => {
            var url = base ? (base + '/' + file) : this._url(file);
            var request = new XMLHttpRequest();
            if (encoding == null) {
                request.responseType = 'arraybuffer';
            }
            request.onload = () => {
                if (request.status == 200) {
                    if (request.responseType == 'arraybuffer') {
                        resolve(new Uint8Array(request.response));
                    }
                    else {
                        resolve(request.responseText);
                    }
                }
                else {
                    reject(request.status);
                }
            };
            request.onerror = () => {
                reject(request.status);
            };
            request.open('GET', url, true);
            request.send();
        });
    }

    openURL(url) {
        window.open(url, '_target');
    }

    exception(err, fatal) {
        if (window.ga && this.version && this.version !== '0.0.0') {
            var description = [];
            description.push((err && err.name ? (err.name + ': ') : '') + (err && err.message ? err.message : '(null)'));
            if (err.stack) {
                var match = err.stack.match(/\n {4}at (.*)\((.*)\)/);
                if (match) {
                    description.push(match[1] + '(' + match[2].split('/').pop() + ')');
                }
                else {
                    description.push(err.stack.split('\n').shift());
                }
            }
            window.ga('send', 'exception', {
                exDescription: description.join(' @ '),
                exFatal: fatal,
                appName: this.type,
                appVersion: this.version
            });
        }
    }

    screen(name) {
        if (window.ga && this.version && this.version !== '0.0.0') {
            window.ga('send', 'screenview', {
                screenName: name,
                appName: this.type,
                appVersion: this.version
            });
        }
    }

    event(category, action, label, value) {
        if (window.ga && this.version && this.version !== '0.0.0') {
            window.ga('send', 'event', {
                eventCategory: category,
                eventAction: action,
                eventLabel: label,
                eventValue: value,
                appName: this.type,
                appVersion: this.version
            });
        }
    }

    _url(file) {
        var url = file;
        if (window && window.location && window.location.href) {
            var location = window.location.href.split('?').shift();
            if (location.endsWith('.html')) {
                location = location.split('/').slice(0, -1).join('/');
            }
            if (location.endsWith('/')) {
                location = location.slice(0, -1);
            }
            url = location + '/' + file;
        }
        return url;
    }

    _getQueryParameter(name) {
        var url = window.location.href;
        name = name.replace(/[[\]]/g, "\\$&");
        var regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)");
        var results = regex.exec(url);
        if (!results) {
            return null;
        }
        if (!results[2]) {
            return '';
        }
        return decodeURIComponent(results[2].replace(/\+/g, " "));
    }

    _openModel(url, identifier) {
        this._view.show('Spinner');
        var request = new XMLHttpRequest();
        request.responseType = 'arraybuffer';
        request.onload = () => {
            if (request.status == 200) {
                var buffer = new Uint8Array(request.response);
                var context = new BrowserContext(this, url, identifier, buffer);
                this._view.open(context).then(() => {
                    this.document.title = identifier || url.split('/').pop();
                }).catch((error) => {
                    if (error) {
                        this.exception(error, false);
                        this.error(error.name, error.message);
                    }
                });
            }
            else {
                this.error('Model load request failed.', request.status);
            }
        };
        request.onerror = () => {
            this.error('Error while requesting model.', request.status);
        };
        request.open('GET', url + ((/\?/).test(url) ? "&" : "?") + (new Date()).getTime(), true);
        request.send();
    }

    _open(file, files) {
        this._view.show('Spinner');
        var context = new BrowserFileContext(file, files);
        context.open().then(() => {
            return this._view.open(context).then((model) => {
                this._view.show(null);
                this.document.title = files[0].name;
                return model;
            });
        }).catch((error) => {
            this._view.show(null);
            this.exception(error, false);
            this.error(error.name, error.message);
        });
    }

    _openGist(gist) {
        this._view.show('Spinner');
        var url = 'https://api.github.com/gists/' + gist;
        var request = new XMLHttpRequest();
        request.onload = () => {
            var identifier = null;
            var buffer = null;
            var json = JSON.parse(request.response);
            if (json.message) {
                this.error('Error while loading Gist.', json.message);
                return;
            }
            if (json.files) {
                for (var key of Object.keys(json.files)) {
                    var file = json.files[key];
                    identifier = file.filename;
                    var extension = identifier.split('.').pop().toLowerCase();
                    if (extension == 'json' || extension == 'pbtxt' || extension == 'prototxt') {
                        var encoder = new TextEncoder();
                        buffer = encoder.encode(file.content);
                    }
                }
            }
            if (buffer == null || identifier == null) {
                this.error('Error while loading Gist.', 'Gist does not contain model file.');
                return;
            }
            var context = new BrowserContext(this, '', identifier, buffer);
            this._view.open(context).then(() => {
                this.document.title = identifier;
            }).catch((error) => {
                if (error) {
                    this.exception(error, false);
                    this.error(error.name, error.message);
                }
            });
        };
        request.onerror = () => {
            this.error('Error while requesting Gist.', request.status);
        };
        request.open('GET', url, true);
        request.send();
    }
};

if (typeof TextDecoder === "undefined") {
    TextDecoder = function TextDecoder(encoding) {
        this._encoding = encoding;
    };
    TextDecoder.prototype.decode = function decode(buffer) {
        var result = '';
        var length = buffer.length;
        var i = 0;
        switch (this._encoding) {
            case 'utf-8':
                while (i < length) {
                    var c = buffer[i++];
                    switch(c >> 4)
                    { 
                        case 0: case 1: case 2: case 3: case 4: case 5: case 6: case 7:
                            result += String.fromCharCode(c);
                            break;
                        case 12: case 13:
                            c2 = buffer[i++];
                            result += String.fromCharCode(((c & 0x1F) << 6) | (c2 & 0x3F));
                            break;
                        case 14:
                            var c2 = buffer[i++];
                            var c3 = buffer[i++];
                            result += String.fromCharCode(((c & 0x0F) << 12) | ((c2 & 0x3F) << 6) | ((c3 & 0x3F) << 0));
                            break;
                    }
                }
                break;
            case 'ascii':
                while (i < length) {
                    result += String.fromCharCode(buffer[i++]);
                }
                break;
        }
        return result;
    };
}

if (typeof TextEncoder === "undefined") {
    TextEncoder = function TextEncoder() {
    };
    TextEncoder.prototype.encode = function encode(str) {
        "use strict";
        var length = str.length, resPos = -1;
        var resArr = typeof Uint8Array === "undefined" ? new Array(length * 2) : new Uint8Array(length * 3);
        for (var point = 0, nextcode = 0, i = 0; i !== length; ) {
            point = str.charCodeAt(i);
            i += 1;
            if (point >= 0xD800 && point <= 0xDBFF) {
                if (i === length) {
                    resArr[resPos += 1] = 0xef; resArr[resPos += 1] = 0xbf;
                    resArr[resPos += 1] = 0xbd; break;
                }
                nextcode = str.charCodeAt(i);
                if (nextcode >= 0xDC00 && nextcode <= 0xDFFF) {
                    point = (point - 0xD800) * 0x400 + nextcode - 0xDC00 + 0x10000;
                    i += 1;
                    if (point > 0xffff) {
                        resArr[resPos += 1] = (0x1e<<3) | (point>>>18);
                        resArr[resPos += 1] = (0x2<<6) | ((point>>>12)&0x3f);
                        resArr[resPos += 1] = (0x2<<6) | ((point>>>6)&0x3f);
                        resArr[resPos += 1] = (0x2<<6) | (point&0x3f);
                        continue;
                    }
                } else {
                    resArr[resPos += 1] = 0xef; resArr[resPos += 1] = 0xbf;
                    resArr[resPos += 1] = 0xbd; continue;
                }
            }
            if (point <= 0x007f) {
                resArr[resPos += 1] = (0x0<<7) | point;
            } else if (point <= 0x07ff) {
                resArr[resPos += 1] = (0x6<<5) | (point>>>6);
                resArr[resPos += 1] = (0x2<<6) | (point&0x3f);
            } else {
                resArr[resPos += 1] = (0xe<<4) | (point>>>12);
                resArr[resPos += 1] = (0x2<<6) | ((point>>>6)&0x3f);
                resArr[resPos += 1] = (0x2<<6) | (point&0x3f);
            }
        }
        if (typeof Uint8Array!=="undefined") {
            return new Uint8Array(resArr.buffer.slice(0, resPos+1));
        }
        else {
            return resArr.length === resPos + 1 ? resArr : resArr.slice(0, resPos + 1);
        }
    };
    TextEncoder.prototype.toString = function() { 
        return "[object TextEncoder]";
    };
    try {
        Object.defineProperty(TextEncoder.prototype,"encoding", {
            get:function() {
                if (Object.prototype.isPrototypeOf.call(TextEncoder.prototype, this)) {
                    return"utf-8";
                }
                else {
                    throw TypeError("Illegal invocation");
                }
            }
        });
    }
    catch (e) {
        TextEncoder.prototype.encoding = "utf-8";
    }
    if (typeof Symbol !== "undefined") {
        TextEncoder.prototype[Symbol.toStringTag] = "TextEncoder";
    }
}

if (!HTMLCanvasElement.prototype.toBlob) {
    HTMLCanvasElement.prototype.toBlob = function(callback, type, quality) {
        var canvas = this;
        setTimeout(function() {
            var data = atob(canvas.toDataURL(type, quality).split(',')[1]);
            var length = data.length;
            var buffer = new Uint8Array(length);
            for (var i = 0; i < length; i++) {
                buffer[i] = data.charCodeAt(i);
            }
            callback(new Blob([ buffer ], { type: type || 'image/png' }));
        });
    };
}

host.Dropdown = class {

    constructor(document, button, dropdown) {
        this._document = document;
        this._dropdown = document.getElementById(dropdown);
        this._button = document.getElementById(button);
        this._items = [];
        this._apple = /(Mac|iPhone|iPod|iPad)/i.test(navigator.platform);
        this._acceleratorMap = {}; 
        window.addEventListener('keydown', (e) => {
            var code = e.keyCode;
            code |= ((e.ctrlKey && !this._apple) || (e.metaKey && this._apple)) ? 0x0400 : 0; 
            code |= e.altKey ? 0x0200 : 0;
            code |= e.shiftKey ? 0x0100 : 0;
            if (code == 0x001b) { // Escape
                this.close();
                return;
            }
            var item = this._acceleratorMap[code.toString()];
            if (item) {
                item.click();
                e.preventDefault();
            }
        });
        this._document.body.addEventListener('click', (e) => {
            if (!this._button.contains(e.target)) {
                this.close();
            }
        });
    }

    add(item) {
        var accelerator = item.accelerator;
        if (accelerator) {
            var cmdOrCtrl = false;
            var alt = false;
            var shift = false;
            var key = '';
            for (var part of item.accelerator.split('+')) {
                switch (part) {
                    case 'CmdOrCtrl': cmdOrCtrl = true; break;
                    case 'Alt': alt = true; break;
                    case 'Shift': shift = true; break;
                    default: key = part; break;
                }
            }
            if (key !== '') {
                item.accelerator = {};
                item.accelerator.text = '';
                if (this._apple) {
                    item.accelerator.text += alt ? '&#x2325;' : '';
                    item.accelerator.text += shift ? '&#x21e7;' : '';
                    item.accelerator.text += cmdOrCtrl ? '&#x2318;' : '';
                    var keyTable = { 'Enter': '&#x23ce;', 'Up': '&#x2191;', 'Down': '&#x2193;', 'Backspace': '&#x232B;' };
                    item.accelerator.text += keyTable[key] ? keyTable[key] : key;
                }
                else {
                    var list = [];
                    if (cmdOrCtrl) {
                        list.push('Ctrl');
                    }
                    if (alt) {
                        list.push('Alt');
                    }
                    if (shift) {
                        list.push('Shift');
                    }
                    list.push(key);
                    item.accelerator.text = list.join('+');
                }
                var code = 0;
                switch (key) {
                    case 'Backspace': code = 0x08; break;
                    case 'Enter': code = 0x0D; break;
                    case 'Up': code = 0x26; break;
                    case 'Down': code = 0x28; break;
                    default: code = key.charCodeAt(0); break;
                }
                code |= cmdOrCtrl ? 0x0400 : 0;
                code |= alt ? 0x0200 : 0;
                code |= shift ? 0x0100 : 0;
                this._acceleratorMap[code.toString()] = item;
            }
        }
        this._items.push(item);
    }

    toggle() {

        if (this._dropdown.style.display === 'block') {
            this.close();
            return;
        }

        while (this._dropdown.lastChild) {
            this._dropdown.removeChild(this._dropdown.lastChild);
        }

        for (var item of this._items) {
            if (Object.keys(item).length > 0) {
                var button = this._document.createElement('button');
                button.innerText = (typeof item.label == 'function') ? item.label() : item.label;
                button.addEventListener('click', item.click);
                button.addEventListener('click', () => this.close());
                this._dropdown.appendChild(button);
                if (item.accelerator) {
                    var accelerator = this._document.createElement('span');
                    accelerator.style.float = 'right';
                    accelerator.innerHTML = item.accelerator.text;
                    button.appendChild(accelerator);
                }
            }
            else {
                var separator = this._document.createElement('div');
                separator.setAttribute('class', 'separator');
                this._dropdown.appendChild(separator);
            }
        }

        this._dropdown.style.display = 'block';
    }

    close() {
        this._dropdown.style.display = 'none';
    }
}


class BrowserFileContext {

    constructor(file, blobs) {
        this._file = file;
        this._blobs = {};
        for (var blob of blobs) {
            this._blobs[blob.name] = blob;
        }
    }

    get identifier() {
        return this._file.name;
    }

    get buffer() {
        return this._buffer;
    }

    open() {
        return this.request(this._file.name, null).then((data) => {
            this._buffer = data;
        });
    }

    request(file, encoding) {
        var blob = this._blobs[file];
        if (!blob) {
            return Promise.reject(new Error("File not found '" + file + "'."));
        }
        return new Promise((resolve, reject) => {
            var reader = new FileReader();
            reader.onload = (e) => {
                resolve(encoding ? e.target.result : new Uint8Array(e.target.result));
            };
            reader.onerror = (e) => {
                e = e || window.event;
                var message = '';
                switch(e.target.error.code) {
                    case e.target.error.NOT_FOUND_ERR:
                        message = "File not found '" + file + "'.";
                        break;
                    case e.target.error.NOT_READABLE_ERR:
                        message = "File not readable '" + file + "'.";
                        break;
                    case e.target.error.SECURITY_ERR:
                        message = "File access denied '" + file + "'.";
                        break;
                    default:
                        message = "File read '" + e.target.error.code.toString() + "' error '" + file + "'.";
                        break;
                }
                reject(new Error(message));
            };
            if (encoding === 'utf-8') {
                reader.readAsText(blob, encoding);
            }
            else {
                reader.readAsArrayBuffer(blob);
            }
        });
    }
}

class BrowserContext {

    constructor(host, url, identifier, buffer) {
        this._host = host;
        this._buffer = buffer;
        if (identifier) {
            this._identifier = identifier;
            this._base = url;
            if (this._base.endsWith('/')) {
                this._base.substring(0, this._base.length - 1);
            }
        }
        else {
            url = url.split('/');
            this._identifier = url.pop();
            this._base = url.join('/');
        }
    }

    request(file, encoding) {
        return this._host.request(this._base, file, encoding);
    }

    get identifier() {
        return this._identifier;
    }

    get buffer() {
        return this._buffer;
    }
}

window.__view__ = new view.View(new host.BrowserHost());
